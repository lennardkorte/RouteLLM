import abc
import functools
import random

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import logging
import torch.nn.functional as F

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
from transformers import AutoConfig

from routellm.routers.causal_llm.configs import RouterModelConfig
from routellm.routers.causal_llm.llm_utils import (
    load_prompt_format,
    to_openai_api_messages,
)
from routellm.routers.causal_llm.model import CausalLLMClassifier
from routellm.routers.matrix_factorization.model import MODEL_IDS, MFModel
from routellm.routers.similarity_weighted.utils import (
    OPENAI_CLIENT,
    compute_elo_mle_with_tie,
    compute_tiers,
    preprocess_battles,
)

def no_parallel(cls):
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    NO_PARALLEL = False

    # Returns a float between 0 and 1 representing the value used to route to models, conventionally the winrate of the strong model.
    # If this value is >= the user defined cutoff, the router will route to the strong model, otherwise, it will route to the weak model.
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        return NAME_TO_CLS[self.__class__]


@no_parallel
class CausalLLMRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        score_threshold=4,
        special_tokens=["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"],
        num_outputs=5,
        model_type="causal",
        model_id="meta-llama/Meta-Llama-3-8B",
        flash_attention_2=False,
    ):
        model_config = RouterModelConfig(
            model_id=model_id,
            model_type=model_type,
            flash_attention_2=flash_attention_2,
            special_tokens=special_tokens,
            num_outputs=num_outputs,
        )
        prompt_format = load_prompt_format(model_config.model_id)
        self.router_model = CausalLLMClassifier(
            config=model_config,
            ckpt_local_path=checkpoint_path,
            score_threshold=score_threshold,
            prompt_format=prompt_format,
            prompt_field="messages",
            additional_fields=[],
            use_last_turn=True,
        )
        system_message = hf_hub_download(
            repo_id=checkpoint_path, filename="system_ft_v5.txt"
        )
        classifier_message = hf_hub_download(
            repo_id=checkpoint_path, filename="classifier_ft_v5.txt"
        )
        with open(system_message, "r") as pr:
            system_message = pr.read()
        with open(classifier_message, "r") as pr:
            classifier_message = pr.read()
        self.to_openai_messages = functools.partial(
            to_openai_api_messages, system_message, classifier_message
        )

    def calculate_strong_win_rate(self, prompt):
        input = {}
        input["messages"] = self.to_openai_messages([prompt])
        output = self.router_model(input)
        if output is None:
            # Route to strong model if output is invalid
            return 1
        else:
            return 1 - output["binary_prob"]


@no_parallel
class BERTRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        num_labels=3,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    def calculate_strong_win_rate(self, prompt):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.numpy()[0]

        exp_scores = np.exp(logits - np.max(logits))
        softmax_scores = exp_scores / np.sum(exp_scores)

        # Compute prob of label 1 and 2 (tie, tier 2 wins)
        binary_prob = np.sum(softmax_scores[-2:])
        return 1 - binary_prob


class SWRankingRouter(Router):
    def __init__(
        self,
        arena_battle_datasets,
        arena_embedding_datasets,
        # This is the model pair for Elo calculations at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        num_tiers=10,
    ):
        self.strong_model = strong_model
        self.weak_model = weak_model

        self.arena_df = concatenate_datasets(
            [load_dataset(dataset, split="train") for dataset in arena_battle_datasets]
        ).to_pandas()
        self.arena_df = preprocess_battles(self.arena_df)

        embeddings = [
            np.array(load_dataset(dataset, split="train").to_dict()["embeddings"])
            for dataset in arena_embedding_datasets
        ]
        self.arena_conv_embedding = np.concatenate(embeddings)
        self.embedding_model = "text-embedding-3-small"

        assert len(self.arena_df) == len(
            self.arena_conv_embedding
        ), "Number of battle embeddings is mismatched to data"

        model_ratings = compute_elo_mle_with_tie(self.arena_df)
        self.model2tier = compute_tiers(model_ratings, num_tiers=num_tiers)

        self.arena_df["model_a"] = self.arena_df["model_a"].apply(
            lambda x: self.model2tier[x]
        )
        self.arena_df["model_b"] = self.arena_df["model_b"].apply(
            lambda x: self.model2tier[x]
        )

    def get_weightings(self, similarities):
        max_sim = np.max(similarities)
        return 10 * 10 ** (similarities / max_sim)

    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        prompt_emb = (
            (
                OPENAI_CLIENT.embeddings.create(
                    input=[prompt], model=self.embedding_model
                )
            )
            .data[0]
            .embedding
        )
        similarities = np.dot(self.arena_conv_embedding, prompt_emb) / (
            np.linalg.norm(self.arena_conv_embedding, axis=1)
            * np.linalg.norm(prompt_emb)
        )

        weightings = self.get_weightings(similarities)
        res = compute_elo_mle_with_tie(self.arena_df, sample_weight=weightings)

        weak_score, strong_score = (
            res[self.model2tier[self.weak_model]],
            res[self.model2tier[self.strong_model]],
        )
        weak_winrate = 1 / (1 + 10 ** ((strong_score - weak_score) / 400))
        strong_winrate = 1 - weak_winrate

        # If the expected strong winrate is greater than the threshold, use strong
        return strong_winrate


@no_parallel
class MatrixFactorizationRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        # This is the model pair for scoring at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        hidden_size=128,
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MFModel.from_pretrained(
            checkpoint_path,
            dim=hidden_size,
            num_models=num_models,
            text_dim=text_dim,
            num_classes=num_classes,
            use_proj=use_proj,
        )
        self.model = self.model.eval().to(device)
        self.strong_model_id = MODEL_IDS[strong_model]
        self.weak_model_id = MODEL_IDS[weak_model]

    def calculate_strong_win_rate(self, prompt):
        winrate = self.model.pred_win_rate(
            self.strong_model_id, self.weak_model_id, prompt
        )
        return winrate


# Parallelism makes the randomness non deterministic
@no_parallel
class RandomRouter(Router):
    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        del prompt
        return random.uniform(0, 1)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=5, nhead=8, nlayers=5):
        super(TransformerClassifier, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.transformer_encoder(x.unsqueeze(1))  # Add sequence dimension
        x = torch.relu(self.fc1(x[:, -1, :]))  # Use the last token
        return self.fc2(x)

# Load the model
model_path = "daparasyte/prompt_complexity_classifier_1"
config = AutoConfig.from_pretrained(model_path)
model = TransformerClassifier(input_dim=config.hidden_size)

# Load weights
model_weights_path = "/content/best_model.pt"  # Path to full model file
model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))

# Save state_dict
torch.save(model.state_dict(), "best_model_state_dict.pt")
print("Model state_dict saved successfully!")

class CostSensitiveRouter(Router):
    def __init__(self, model_path="daparasyte/prompt_complexity_classifier_1", strong_model_cost=1.5, weak_model_cost=1.0):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.strong_model_cost = strong_model_cost
        self.weak_model_cost = weak_model_cost

        # Load the embedding model
        self.embedding_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

        # Initialize the complexity model
        config = AutoConfig.from_pretrained(model_path)
        self.prompt_complexity_model = TransformerClassifier(input_dim=config.hidden_size)

        # Load the state_dict into the model
        model_weights_path = "best_model_state_dict.pt"
        self.prompt_complexity_model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.prompt_complexity_model.to(self.device)
        self.prompt_complexity_model.eval()

    @staticmethod
    def calculate_win_probabilities(prompt, embedding_model, complexity_model, device):
        # Encode the prompt into embeddings
        embedding = embedding_model.encode(prompt, convert_to_tensor=True).to(torch.float32).to(device)

        # Directly use the preloaded complexity_model
        with torch.no_grad():
            embedding = embedding.unsqueeze(0)  # Add batch dimension
            output = complexity_model(embedding)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

        # Calculate strong and weak win probabilities
        strong_win = probabilities[0] + probabilities[1] + probabilities[2] / 2
        weak_win = probabilities[2] / 2 + probabilities[3] + probabilities[4]
        return strong_win, weak_win


    def calculate_strong_win_rate(self, prompt):
        strong_win, weak_win = self.calculate_win_probabilities(
            prompt, self.embedding_model, self.prompt_complexity_model, self.device
        )
        print(f"Prompt: {prompt}")
        print(f"Strong Win Probability: {strong_win}, Weak Win Probability: {weak_win}")
        return strong_win


    def route(self, prompt, routed_pair):
        win_rate = self.calculate_strong_win_rate(prompt)

        # Determine prompt complexity and adjust the threshold dynamically
        if win_rate < 0.3:
            threshold = 0.7
        elif win_rate < 0.6:
            threshold = 0.5
        else:
            threshold = 0.4

        # Adjust for cost sensitivity
        cost_factor = self.weak_model_cost / self.strong_model_cost
        threshold *= cost_factor

        print(f"Prompt: {prompt}")
        print(f"Win rate: {win_rate}")
        print(f"Dynamic threshold (after cost adjustment): {threshold}")

        return routed_pair.strong if win_rate >= threshold else routed_pair.weak

class BayesianOptimisationRouter(CostSensitiveRouter):
    def __init__(
        self,
        low_threshold=0.1,
        high_threshold=1.0,
        strong_model_cost=1.5,
        weak_model_cost=1.0,
        complexity_scaling=0.5,
    ):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.strong_model_cost = strong_model_cost
        self.weak_model_cost = weak_model_cost
        self.complexity_scaling = complexity_scaling


    def calculate_prompt_complexity(self, prompt):
        """
        Calculate the complexity score of the prompt using the complexity model.
        This method follows the logic used in CostSensitiveRouter.
        """
        embedding = self.embedding_model.encode(prompt, convert_to_tensor=True).to(torch.float32).to(self.device)

        # Use the prompt complexity model to generate output probabilities
        with torch.no_grad():
            embedding = embedding.unsqueeze(0)  # Add batch dimension
            output = self.prompt_complexity_model(embedding)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

        # The complexity score can be derived from the probabilities
        # Adjust this based on your specific interpretation of complexity
        return probabilities[2]  # Example: Using the third class probability as complexity

    def route(self, prompt, routed_pair):
        # Scale complexity and calculate a dynamic threshold
        complexity_score = self.calculate_prompt_complexity(prompt)
        dynamic_threshold = self.low_threshold + (complexity_score * self.complexity_scaling)

        # Apply cost sensitivity
        cost_factor = self.weak_model_cost / self.strong_model_cost
        dynamic_threshold *= cost_factor

        print(f"Dynamic Threshold: {dynamic_threshold}")
        win_rate = self.calculate_strong_win_rate(prompt)

        return routed_pair.strong if win_rate >= dynamic_threshold else routed_pair.weak

ROUTER_CLS = {
    "random": RandomRouter,
    "mf": MatrixFactorizationRouter,
    "causal_llm": CausalLLMRouter,
    "bert": BERTRouter,
    "sw_ranking": SWRankingRouter,
    "cost_sensitive": CostSensitiveRouter,
    "bayesian_optimisation": BayesianOptimisationRouter,
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
