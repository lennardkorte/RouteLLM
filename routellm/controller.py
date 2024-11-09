from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
from litellm import acompletion, completion
from tqdm import tqdm

from routellm.routers.routers import ROUTER_CLS

# Default config for routers augmented using golden label data from GPT-4.
# This is exactly the same as config.example.yaml.
GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "arena_battle_datasets": [
            "lmsys/lmsys-arena-human-preference-55k",
            "routellm/gpt4_judge_battles",
        ],
        "arena_embedding_datasets": [
            "routellm/arena_battles_embeddings",
            "routellm/gpt4_judge_battles_embeddings",
        ],
    },
    "causal_llm": {"checkpoint_path": "routellm/causal_llm_gpt4_augmented"},
    "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"},
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}


class RoutingError(Exception):
    pass


@dataclass
class ModelSet:
    models: list[str]


class Controller:
    def __init__(
        self,
        routers: list[str],
        models: list[str],
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
    ):
        self.model_set = ModelSet(models=models)
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        router_pbar = None
        if progress_bar:
            router_pbar = tqdm(routers)
            tqdm.pandas()

        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            self.routers[router] = ROUTER_CLS[router](**config.get(router, {}))

        # Some Python magic to match the OpenAI Python SDK
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.completion, acreate=self.acompletion
            )
        )

    def _validate_router_threshold(
        self, router: Optional[str], threshold: Optional[float]
    ):
        if router is None or threshold is None:
            raise RoutingError("Router or threshold unspecified.")
        if router not in self.routers:
            raise RoutingError(
                f"Invalid router {router}. Available routers are {list(self.routers.keys())}."
            )
        if not 0 <= threshold <= 1:
            raise RoutingError(
                f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            )

    def _parse_model_name(self, model: str):
        _, router, threshold = model.split("-", 2)
        try:
            threshold = float(threshold)
        except ValueError as e:
            raise RoutingError(f"Threshold {threshold} must be a float.") from e
        if not model.startswith("router"):
            raise RoutingError(
                f"Invalid model {model}. Model name must be of the format 'router-[router name]-[threshold]."
            )
        return router, threshold

    def _get_routed_model_for_completion(self, messages: list, router: str, threshold: float):
        prompt = messages[-1]["content"]
        
        # Obtain sorted list of models with scores
        model_scores = self.routers[router].rank_models_for_prompt(prompt, self.model_set.models)

        # Check if model_scores is empty and raise an error if no models returned
        if not model_scores:
            raise RoutingError("No models returned from rank_models_for_prompt.")

        # Iterate through sorted model_scores, return the first that meets or exceeds the threshold
        for model, score in model_scores:
            if score >= threshold:
                self.model_counts[router][model] += 1  # Track usage for each model
                return model

        # If no model meets the threshold, return the highest scoring model as a fallback
        fallback_model = model_scores[0][0]
        self.model_counts[router][fallback_model] += 1
        return fallback_model

    # Mainly used for evaluations
    def batch_calculate_scores(self, prompts: pd.Series, router: str):
        router_instance = self.routers[router]
        if router_instance.NO_PARALLEL and self.progress_bar:
            return prompts.progress_apply(lambda p: router_instance.rank_models_for_prompt(p, self.model_set.models))
        elif router_instance.NO_PARALLEL:
            return prompts.apply(lambda p: router_instance.rank_models_for_prompt(p, self.model_set.models))
        else:
            return prompts.parallel_apply(lambda p: router_instance.rank_models_for_prompt(p, self.model_set.models))

    def route(self, prompt: str, router: str, threshold: float):
        self._validate_router_threshold(router, threshold)
        return self.routers[router].route(prompt, threshold, self.model_set)
    
    # Matches OpenAI's Chat Completions interface, but also supports optional router and threshold args
    # If model name is present, attempt to parse router and threshold using it, otherwise, use the router and threshold args
    def completion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        return completion(api_base=self.api_base, api_key=self.api_key, **kwargs)

    # Matches OpenAI's Async Chat Completions interface, but also supports optional router and threshold args
    async def acompletion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        kwargs["model"] = self._get_routed_model_for_completion(
            kwargs["messages"], router, threshold
        )
        return await acompletion(api_base=self.api_base, api_key=self.api_key, **kwargs)
