import abc
import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from routellm.controller import Controller
from routellm.routers.routers import Router
from routellm.routers.routers import CostSensitiveRouter


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

pd.options.mode.copy_on_write = True


class Benchmark(abc.ABC):
    """
    Benchmark class for evaluating models.

    Internally, class should handle init and manage own cache (if needed).
    """

    @abc.abstractmethod
    def evaluate(
        self,
        controller: Controller,
        router: str,
        num_results: int,
        overwrite_router_cache: bool,
    ) -> tuple[str, dict[str, int], str]:
        """Takes in a router and threshold and returns a tuple of weighted accuracy, model counts, and number of requests."""
        pass

    @abc.abstractmethod
    def get_optimal_accuracy(self, strong_percent: float) -> float:
        """Takes in % strong model calls and returns the optimal score for the benchmark given these % of calls."""
        pass

    @abc.abstractmethod
    def get_model_accuracy(self, model: str) -> float:
        """Takes in a model name and returns the accuracy of that model on the benchmark."""
        pass


class MMLU(Benchmark):
    def __init__(self, domains, routed_pair, overwrite_cache):
        self.routed_pair = routed_pair
        self.overwrite_cache = overwrite_cache
        self.cache_path = f"{CURRENT_DIR}/mmlu/cache.npy"

        try:
            self.cache = np.load(self.cache_path, allow_pickle=True).item()
        except:
            self.cache = {}

        all_data = pd.DataFrame()
        for domain in tqdm(domains, desc="Loading domain data"):
            all_data = pd.concat(
                [
                    all_data,
                    pd.read_csv(f"{CURRENT_DIR}/mmlu/responses/mmlu_{domain}.csv"),
                ],
                ignore_index=True,
            )
        original_length = len(all_data)

        # Generated using contamination_check.py
        contaminated_prompts = pd.read_json(
            f"{CURRENT_DIR}/mmlu/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()
        self.all_data = all_data[~all_data["prompt"].isin(contaminated_prompts)]
        print(
            f"Remaining {len(self.all_data)}/{original_length} prompts for MMLU after decontamination"
        )

    def evaluate(self, controller, router_instance, num_results, overwrite_router_cache):
        print(f"Router instance type: {type(router_instance)}")  # Debugging: Print router type
        if overwrite_router_cache or type(router_instance).__name__ not in self.cache:
            # Use the router_instance's method to calculate strong win rates
            strong_win_rates = self.all_data["prompt"].apply(
                lambda p: router_instance.calculate_strong_win_rate(p)
            )
            # Cache the computed strong win rates
            self.cache[type(router_instance).__name__] = strong_win_rates
            np.save(self.cache_path, self.cache)
        else:
            # Load cached strong win rates
            strong_win_rates = self.cache[type(router_instance).__name__]

        print(f"Strong win rates calculated for router: {type(router_instance).__name__}")

        # Process thresholds
        _, thresholds = pd.qcut(strong_win_rates, num_results, retbins=True, duplicates="drop")
        self.all_data["strong_win_rates"] = strong_win_rates

        for i, threshold in enumerate(thresholds):
            selection = (
                self.all_data["strong_win_rates"] >= threshold
                if i != len(thresholds) - 1
                else self.all_data["strong_win_rates"] > threshold
            )
            results = np.where(
                selection,
                self.all_data[self.routed_pair.strong],
                self.all_data[self.routed_pair.weak],
            )
            models = np.where(
                selection,
                self.routed_pair.strong,
                self.all_data[self.routed_pair.weak],
            )
            model_counts = Counter(models)
            yield threshold, sum(results) / len(results) * 100, model_counts, len(results)


    def get_optimal_accuracy(self, strong_percent):
        df = self.all_data
        total = len(df)

        strong_calls = total * strong_percent
        weak_correct = len(df[df[self.routed_pair.weak] == True])

        df_sub = df[df[self.routed_pair.weak] == False]
        df_sub = df_sub[df_sub[self.routed_pair.strong] == True]

        strong_bonus = min(strong_calls, len(df_sub))
        opt_correct = weak_correct + strong_bonus
        opt_accuracy = opt_correct / total * 100

        return opt_accuracy

    def get_model_accuracy(self, model):
        df = self.all_data
        return len(df[df[model] == True]) / len(df) * 100


class MTBench(Benchmark):
    def __init__(self, routed_pair, overwrite_cache):
        self.routed_pair = routed_pair

        self.judgements = pd.read_json(
            f"{CURRENT_DIR}/mt_bench/judgements.jsonl", lines=True
        )
        self.questions = pd.read_json(
            f"{CURRENT_DIR}/mt_bench/question.jsonl", lines=True
        )
        contaminated_prompts = pd.read_json(
            f"{CURRENT_DIR}/mt_bench/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()

        self.questions["turn1"] = self.questions["turns"].apply(lambda x: x[0])
        self.questions["turn2"] = self.questions["turns"].apply(lambda x: x[1])
        self.questions = self.questions[
            ~(
                self.questions["turn1"].isin(contaminated_prompts)
                | self.questions["turn2"].isin(contaminated_prompts)
            )
        ]
        print(f"{len(self.questions)} questions for MT bench after decontamination.")

        self.overwrite_cache = overwrite_cache
        self.cache_path = f"{CURRENT_DIR}/mt_bench/cache.npy"

        try:
            self.cache = np.load(self.cache_path, allow_pickle=True).item()
        except:
            print("Error loading MT Bench cache, starting fresh.")
            self.cache = {}

    def evaluate(self, controller, router_instance, num_results, overwrite_router_cache):
        print(f"Router instance type: {type(router_instance)}")  # Debugging router type

        # Cache handling
        if overwrite_router_cache or type(router_instance).__name__ not in self.cache:
            # Calculate strong win rates using the router instance
            strong_win_rates = self.questions["turns"].apply(
                lambda x: router_instance.calculate_strong_win_rate(x[0])  # Use the first turn for routing
            )
            self.cache[type(router_instance).__name__] = strong_win_rates
            np.save(self.cache_path, self.cache)
        else:
            strong_win_rates = self.cache[type(router_instance).__name__]

        print(f"Strong win rates calculated for router: {type(router_instance).__name__}")

        # Process thresholds
        _, thresholds = pd.qcut(strong_win_rates, num_results, retbins=True, duplicates="drop")
        self.questions["strong_win_rates"] = strong_win_rates

        for i, threshold in enumerate(thresholds):
            # Assign routed models based on threshold
            self.questions["routed_model"] = np.where(
                (
                    self.questions["strong_win_rates"] >= threshold
                    if i != len(thresholds) - 1
                    else self.questions["strong_win_rates"] > threshold
                ),
                self.routed_pair.strong,
                self.routed_pair.weak,
            )

            # Merge with judgements to compute scores
            results = self.questions.merge(
                self.judgements,
                left_on=["question_id", "routed_model"],
                right_on=["question_id", "model"],
                how="left",
            )[["question_id", "model", "score"]]

            # Calculate average score
            score = results["score"].mean()

            # Count how many prompts were routed to each model
            model_counts = results["model"].value_counts().to_dict()
            if self.routed_pair.weak not in model_counts:
                model_counts[self.routed_pair.weak] = 0
            if self.routed_pair.strong not in model_counts:
                model_counts[self.routed_pair.strong] = 0

            # Ensure data consistency
            total = len(results)
            assert total == sum(model_counts.values()) == len(self.questions) * 2

            yield threshold, score, model_counts, total

    def get_model_accuracy(self, model):
        questions = self.questions[["question_id"]]
        questions["routed_model"] = model

        results = questions.merge(
            self.judgements,
            left_on=["question_id", "routed_model"],
            right_on=["question_id", "model"],
            how="left",
        )[["question_id", "model", "score"]]

        return results["score"].mean()

    def get_optimal_accuracy(self, strong_percent):
        max_strong_calls = int(len(self.questions) * strong_percent)

        strong_judgements = (
            self.judgements[self.judgements["model"] == self.routed_pair.strong][
                ["question_id", "model", "score"]
            ]
            .groupby(by=["model", "question_id"], as_index=False)
            .mean()
        )

        weak_judgements = (
            self.judgements[self.judgements["model"] == self.routed_pair.weak][
                [
                    "question_id",
                    "model",
                    "score",
                ]
            ]
            .groupby(by=["model", "question_id"], as_index=False)
            .mean()
        )

        combined_judgements = strong_judgements.merge(
            weak_judgements,
            on=["question_id"],
            how="left",
            suffixes=("_strong", "_weak"),
        )
        combined_judgements["diff"] = (
            combined_judgements["score_strong"] - combined_judgements["score_weak"]
        )
        combined_judgements = combined_judgements.sort_values(
            by=["diff"], ascending=False
        ).reset_index(drop=True)

        if len(combined_judgements[combined_judgements["diff"] > 0]) > max_strong_calls:
            combined_judgements.loc[:max_strong_calls, "score_optimal"] = (
                combined_judgements.loc[:max_strong_calls, "score_strong"]
            )
            combined_judgements.loc[max_strong_calls:, "score_optimal"] = (
                combined_judgements.loc[max_strong_calls:, "score_weak"]
            )
        else:
            combined_judgements["score_optimal"] = combined_judgements[
                "score_strong"
            ].where(combined_judgements["diff"] > 0, combined_judgements["score_weak"])

        assert (
            len(strong_judgements) == len(weak_judgements) == len(combined_judgements)
        )

        return combined_judgements["score_optimal"].mean()


class GSM8K(Benchmark):
    def __init__(self, routed_pair, overwrite_cache):
        self.routed_pair = routed_pair
        self.overwrite_cache = overwrite_cache
        self.cache_path = f"{CURRENT_DIR}/gsm8k/cache.npy"

        try:
            self.cache = np.load(self.cache_path, allow_pickle=True).item()
        except:
            self.cache = {}

        all_data = pd.read_csv(f"{CURRENT_DIR}/gsm8k/gsm8k_responses.csv")
        original_len = len(all_data)

        contaminated_prompts = pd.read_json(
            f"{CURRENT_DIR}/gsm8k/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()
        self.all_data = all_data[~all_data["prompt"].isin(contaminated_prompts)]
        print(
            f"{len(self.all_data)}/{original_len} questions for GSM8K after decontamination."
        )

    def evaluate(self, controller, router_instance, num_results, overwrite_router_cache):
        print(f"Router instance type: {type(router_instance)}")
        if overwrite_router_cache or type(router_instance).__name__ not in self.cache:
            strong_win_rates = self.all_data["prompt"].apply(
                lambda p: router_instance.calculate_strong_win_rate(p)
            )
            self.cache[type(router_instance).__name__] = strong_win_rates
            np.save(self.cache_path, self.cache)
        else:
            strong_win_rates = self.cache[type(router_instance).__name__]


        print(f"Strong win rates calculated for router: {type(router_instance).__name__}")

        # Process thresholds
        _, thresholds = pd.qcut(strong_win_rates, num_results, retbins=True, duplicates="drop")
        self.all_data["strong_win_rates"] = strong_win_rates

        for i, threshold in enumerate(thresholds):
            selection = (
                self.all_data["strong_win_rates"] >= threshold
                if i != len(thresholds) - 1
                else self.all_data["strong_win_rates"] > threshold
            )
            results = np.where(
                selection,
                self.all_data[self.routed_pair.strong],
                self.all_data[self.routed_pair.weak],
            )
            models = np.where(selection, self.routed_pair.strong, self.all_data[self.routed_pair.weak])
            model_counts = Counter(models)
            yield threshold, sum(results) / len(results) * 100, model_counts, len(results)


    def get_model_accuracy(self, model):
        df = self.all_data
        return len(df[df[model] == True]) / len(df) * 100

    def get_optimal_accuracy(self, strong_percent):
        df = self.all_data
        total = len(df)

        strong_calls = total * strong_percent
        weak_correct = len(df[df[self.routed_pair.weak] == True])

        df_sub = df[df[self.routed_pair.weak] == False]
        df_sub = df_sub[df_sub[self.routed_pair.strong] == True]

        strong_bonus = min(strong_calls, len(df_sub))
        opt_correct = weak_correct + strong_bonus
        opt_accuracy = opt_correct / total * 100

        return opt_accuracy
