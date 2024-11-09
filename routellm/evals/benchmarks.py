import abc
import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from routellm.controller import Controller
from routellm.routers.routers import Router

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
    def __init__(self, domains, model_set, overwrite_cache):
        self.model_set = model_set  # ModelSet containing multiple models
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

        # Filter out contaminated prompts
        contaminated_prompts = pd.read_json(
            f"{CURRENT_DIR}/mmlu/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()
        self.all_data = all_data[~all_data["prompt"].isin(contaminated_prompts)]
        print(
            f"Remaining {len(self.all_data)}/{original_length} prompts for MMLU after decontamination"
        )

    def evaluate(self, controller, router, num_results, overwrite_router_cache):
        # Calculate or load router scores for each prompt
        if (
            router not in self.cache
            or router in self.overwrite_cache
            or overwrite_router_cache
        ):
            model_scores = controller.batch_calculate_scores(
                prompts=self.all_data["prompt"], router=router
            )
            self.cache[router] = model_scores
            np.save(self.cache_path, self.cache)
        else:
            model_scores = self.cache[router]

        # For each threshold, rank models dynamically for each prompt
        thresholds = np.linspace(0, 1, num_results)
        results_data = []

        for threshold in thresholds:
            selected_models = []
            model_accuracies = []

            for prompt, scores in zip(self.all_data["prompt"], model_scores):
                # Rank models based on score and apply threshold
                ranked_models = sorted(zip(self.model_set.models, scores), key=lambda x: x[1], reverse=True)
                selected_model = next((model for model, score in ranked_models if score >= threshold), ranked_models[0][0])
                selected_models.append(selected_model)

                # Check if selected model was correct
                is_correct = self.all_data.loc[self.all_data["prompt"] == prompt, selected_model].values[0]
                model_accuracies.append(is_correct)

            accuracy = np.mean(model_accuracies) * 100
            model_counts = Counter(selected_models)
            results_data.append({
                "threshold": threshold,
                "accuracy": accuracy,
                "model_counts": model_counts,
                "total": len(model_accuracies),
            })
            yield threshold, accuracy, model_counts, len(model_accuracies)

    def get_optimal_accuracy(self, strong_percent):
        # Calculate optimal accuracy by selecting highest-performing models up to strong_percent
        total = len(self.all_data)
        max_strong_calls = int(total * strong_percent)

        strong_model_correct = {model: self.all_data[model].sum() for model in self.model_set.models}
        sorted_models = sorted(strong_model_correct.items(), key=lambda x: x[1], reverse=True)
        
        # Sum optimal performance of selected top-performing models
        optimal_correct = sum(count for model, count in sorted_models[:max_strong_calls])
        opt_accuracy = optimal_correct / total * 100
        return opt_accuracy

    def get_model_accuracy(self, model):
        # Return the accuracy of a specific model on the MMLU dataset
        return len(self.all_data[self.all_data[model] == True]) / len(self.all_data) * 100

class MTBench(Benchmark):
    def __init__(self, model_set, overwrite_cache):
        self.model_set = model_set  # ModelSet containing multiple models
        self.overwrite_cache = overwrite_cache
        self.cache_path = f"{CURRENT_DIR}/mt_bench/cache.npy"

        # Load judgements and questions
        self.judgements = pd.read_json(f"{CURRENT_DIR}/mt_bench/judgements.jsonl", lines=True)
        self.questions = pd.read_json(f"{CURRENT_DIR}/mt_bench/question.jsonl", lines=True)

        # Filter out contaminated prompts
        contaminated_prompts = pd.read_json(
            f"{CURRENT_DIR}/mt_bench/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()
        self.questions["turn1"] = self.questions["turns"].apply(lambda x: x[0])
        self.questions["turn2"] = self.questions["turns"].apply(lambda x: x[1])
        self.questions = self.questions[
            ~(self.questions["turn1"].isin(contaminated_prompts) | self.questions["turn2"].isin(contaminated_prompts))
        ]
        print(f"{len(self.questions)} questions for MT bench after decontamination.")

        # Load or initialize cache
        try:
            self.cache = np.load(self.cache_path, allow_pickle=True).item()
        except:
            print("Error loading MT Bench cache, starting fresh.")
            self.cache = {}

    def evaluate(self, controller, router, num_results, overwrite_router_cache):
        # Get or calculate router scores for each prompt
        if (
            router not in self.cache
            or router in self.overwrite_cache
            or overwrite_router_cache
        ):
            model_scores = controller.batch_calculate_scores(
                prompts=self.questions["turns"].apply(lambda x: x[0]),  # First turn only
                router=router,
            )
            self.cache[router] = model_scores
            np.save(self.cache_path, self.cache)
        else:
            model_scores = self.cache[router]

        thresholds = np.linspace(0, 1, num_results)
        for threshold in thresholds:
            selected_models = []
            scores = []

            # Select the best model per prompt based on the threshold
            for question_id, scores_for_question in zip(self.questions["question_id"], model_scores):
                ranked_models = sorted(zip(self.model_set.models, scores_for_question), key=lambda x: x[1], reverse=True)
                selected_model = next((model for model, score in ranked_models if score >= threshold), ranked_models[0][0])
                selected_models.append(selected_model)

                # Get score for the selected model on this question
                score = self.judgements.loc[
                    (self.judgements["question_id"] == question_id) & (self.judgements["model"] == selected_model),
                    "score",
                ]
                if not score.empty:
                    scores.append(score.iloc[0])

            # Calculate mean score and model distribution
            mean_score = np.mean(scores) if scores else 0
            model_counts = Counter(selected_models)
            yield threshold, mean_score, model_counts, len(selected_models)

    def get_model_accuracy(self, model):
        # Calculate the mean score for a specific model across all questions
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

        # Get mean score per model across questions
        model_performance = {}
        for model in self.model_set.models:
            model_performance[model] = (
                self.judgements[self.judgements["model"] == model]["score"].mean()
            )

        # Sort models by their mean scores
        sorted_models = sorted(model_performance.items(), key=lambda x: x[1], reverse=True)

        # Sum the scores of the top-performing models based on the strong percentage
        top_scores = [score for _, score in sorted_models[:max_strong_calls]]
        optimal_score = sum(top_scores) / len(self.questions)

        return optimal_score

class GSM8K(Benchmark):
    def __init__(self, model_set, overwrite_cache):
        self.model_set = model_set  # ModelSet with multiple models
        self.overwrite_cache = overwrite_cache
        self.cache_path = f"{CURRENT_DIR}/gsm8k/cache.npy"

        try:
            self.cache = np.load(self.cache_path, allow_pickle=True).item()
        except:
            self.cache = {}

        all_data = pd.read_csv(f"{CURRENT_DIR}/gsm8k/gsm8k_responses.csv")
        original_len = len(all_data)

        # Remove contaminated prompts
        contaminated_prompts = pd.read_json(
            f"{CURRENT_DIR}/gsm8k/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()
        self.all_data = all_data[~all_data["prompt"].isin(contaminated_prompts)]
        print(
            f"{len(self.all_data)}/{original_len} questions for GSM8K after decontamination."
        )

    def evaluate(self, controller, router, num_results, overwrite_router_cache):
        # Load or compute router scores for each prompt
        if (
            router not in self.cache
            or router in self.overwrite_cache
            or overwrite_router_cache
        ):
            model_scores = controller.batch_calculate_scores(
                prompts=self.all_data["prompt"], router=router
            )
            self.cache[router] = model_scores
            np.save(self.cache_path, self.cache)
        else:
            model_scores = self.cache[router]

        # Define thresholds to split the score range for evaluation
        thresholds = np.linspace(0, 1, num_results)
        results_data = []

        for threshold in thresholds:
            selected_models = []
            model_accuracies = []

            for prompt, scores in zip(self.all_data["prompt"], model_scores):
                # Rank models by scores and apply threshold
                ranked_models = sorted(zip(self.model_set.models, scores), key=lambda x: x[1], reverse=True)
                selected_model = next((model for model, score in ranked_models if score >= threshold), ranked_models[0][0])
                selected_models.append(selected_model)

                # Check if the selected model answered correctly
                is_correct = self.all_data.loc[self.all_data["prompt"] == prompt, selected_model].values[0]
                model_accuracies.append(is_correct)

            # Calculate accuracy and model counts for the threshold
            accuracy = np.mean(model_accuracies) * 100
            model_counts = Counter(selected_models)
            results_data.append({
                "threshold": threshold,
                "accuracy": accuracy,
                "model_counts": model_counts,
                "total": len(model_accuracies),
            })
            yield threshold, accuracy, model_counts, len(model_accuracies)

    def get_model_accuracy(self, model):
        # Calculate accuracy for a specific model
        df = self.all_data
        return len(df[df[model] == True]) / len(df) * 100

    def get_optimal_accuracy(self, strong_percent):
        # Calculate optimal accuracy by selecting best-performing models up to strong_percent
        df = self.all_data
        total = len(df)
        max_strong_calls = int(total * strong_percent)

        # Count correct answers for each model
        model_correct_counts = {model: df[model].sum() for model in self.model_set.models}
        sorted_models = sorted(model_correct_counts.items(), key=lambda x: x[1], reverse=True)

        # Sum the correct answers from top-performing models
        optimal_correct = sum(count for model, count in sorted_models[:max_strong_calls])
        opt_accuracy = optimal_correct / total * 100
        return opt_accuracy