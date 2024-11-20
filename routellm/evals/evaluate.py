import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml
import json
import argparse
from pandarallel import pandarallel

from routellm.controller import Controller
from routellm.evals.benchmarks import GSM8K, MMLU, MTBench
from routellm.evals.mmlu.domains import ALL_MMLU_DOMAINS
from routellm.routers.routers import CostSensitiveRouter, ROUTER_CLS
from routellm.routers.routers import BayesianOptimisationRouter



os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_results(
    df_router_result, benchmark, benchmark_name, routed_pair, output, plot_optimal=False
):
    plt.figure(figsize=(6, 5))
    for method in df_router_result["method"].unique():
        df_per_method = df_router_result[
            df_router_result["method"] == method
        ].sort_values(by=["strong_percentage"])

        plt.plot(
            df_per_method["strong_percentage"],
            df_per_method["accuracy"],
            label=f"{method}",
            marker=".",
            linestyle="-",
        )

    weak_accuracy = benchmark.get_model_accuracy(routed_pair.weak)
    print(f"{routed_pair.weak} score: {weak_accuracy}")

    strong_accuracy = benchmark.get_model_accuracy(routed_pair.strong)
    print(f"{routed_pair.strong} score: {strong_accuracy}")

    plt.axhline(
        y=weak_accuracy,
        color="grey",
        linestyle="--",
        label=routed_pair.weak,
    )
    plt.axhline(
        y=strong_accuracy,
        color="red",
        linestyle="--",
        label=routed_pair.strong,
    )

    if plot_optimal:
        optimal_accs = []
        optimal_range = range(0, 101, 10)
        for strong_percent in optimal_range:
            optimal_accs.append(benchmark.get_optimal_accuracy(strong_percent / 100))
        plt.plot(
            optimal_range,
            optimal_accs,
            label="Optimal",
            marker="x",
            linestyle="-",
        )

    plt.xlabel("Strong Model Calls (%)")
    plt.ylabel("Performance")
    plt.title(f"Router Performance ({benchmark_name})")
    plt.legend()


    file_name = f"/content/RouteLLM/bo_{benchmark_name}.png"

    print(f"Saving plot to {file_name}")
    plt.savefig(file_name, bbox_inches="tight")

    def pct_call_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])
        pct_calls = []

        for pct in [0.2, 0.5, 0.8]:
            pct_call = np.interp(
                pct * (strong_accuracy - weak_accuracy) + weak_accuracy,
                df_per_method["accuracy"],
                df_per_method["strong_percentage"],
            )
            pct_calls.append(f"{pct_call:.2f}%")

        return pd.Series(pct_calls)

    def auc_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])
        return np.trapz(
            df_per_method["accuracy"], df_per_method["strong_percentage"] / 100
        )

    def apgr_metric(row):
        df_per_method = df_router_result[
            df_router_result["method"] == row["method"]
        ].sort_values(by=["strong_percentage"])

        weak_auc = np.zeros([len(df_per_method)], dtype=float)
        weak_auc.fill(weak_accuracy)
        weak_auc = np.trapz(weak_auc, df_per_method["strong_percentage"] / 100)

        strong_auc = np.zeros([len(df_per_method)], dtype=float)
        strong_auc.fill(strong_accuracy)
        strong_auc = np.trapz(strong_auc, df_per_method["strong_percentage"] / 100)

        return (row["AUC"] - weak_auc) / (strong_auc - weak_auc)

    metrics = pd.DataFrame({"method": df_router_result["method"].unique()})
    metrics[["20% qual", "50% qual", "80% qual"]] = metrics.apply(
        pct_call_metric, axis=1
    )
    metrics["AUC"] = metrics.apply(auc_metric, axis=1)
    metrics["APGR"] = metrics.apply(apgr_metric, axis=1)
    metrics = metrics.sort_values(by=["APGR"], ascending=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("Metrics:\n", metrics)


def pretty_print_results(threshold, accuracy, model_counts, total):
    header = (
        "=" * 15
        + f" {router} with threshold {threshold} on {args.benchmark} "
        + "=" * 15
    )
    print("\n" + header)
    print("Average accuracy: {:.3f}".format(accuracy))
    print(f"Model counts: {', '.join([f'{k}: {v}' for k, v in model_counts.items()])}")
    print(
        f"Model %: {', '.join([f'{k}: {v / total * 100:.3f}%' for k, v in model_counts.items()])}"
    )
    print("=" * len(header) + "\n")


# For bayesian optimisation, load best parameters from file
with open("best_params.json", "r") as f:
    best_params = json.load(f)

print("Loaded best parameters:", best_params)


# Instantiate the BayesianOptimisationRouter
router_instance = BayesianOptimisationRouter(
    low_threshold=best_params["low_threshold"],
    high_threshold=best_params["high_threshold"],
    strong_model_cost=best_params["strong_model_cost"],
    weak_model_cost=best_params["weak_model_cost"],
    complexity_scaling=best_params["complexity_scaling"]
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate routers on various benchmarks."
    )
    parser.add_argument(
        "--routers",
        nargs="+",
        type=str,
        default=["random"],
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=[
            "mmlu",
            "mt-bench",
            "gsm8k",
        ],
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--overwrite-cache",
        nargs="*",
        type=str,
        default=[],
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=psutil.cpu_count(logical=False),
        help="Number of cores to use, all by default.",
    )
    parser.add_argument("--strong-model", type=str, default="gpt-4-1106-preview")
    parser.add_argument(
        "--weak-model",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num-results", type=int, default=10)
    parser.add_argument("--random-iters", type=int, default=10)

    args = parser.parse_args()
    print(args)

    pandarallel.initialize(progress_bar=True, nb_workers=args.parallel)
    controller = Controller(
        routers=args.routers,
        config=yaml.safe_load(open(args.config, "r")) if args.config else None,
        strong_model=args.strong_model,
        weak_model=args.weak_model,
        progress_bar=True,
    )

    if args.benchmark == "mmlu":
        print("Running eval for full MMLU.")
        mmlu_domains = ALL_MMLU_DOMAINS
        benchmark = MMLU(mmlu_domains, controller.model_pair, args.overwrite_cache)
    elif args.benchmark == "mt-bench":
        print("Running eval for MT Bench.")
        benchmark = MTBench(controller.model_pair, args.overwrite_cache)
    elif args.benchmark == "gsm8k":
        print("Running eval for GSM8k.")
        benchmark = GSM8K(controller.model_pair, args.overwrite_cache)
    else:
        raise ValueError(f"Invalid benchmark {args.benchmark}")

    print(f"Evaluating routers on {args.benchmark}...")

    all_results = pd.DataFrame()
    random.seed(0)

    # Routers to evaluate: random, cost_sensitive, and bayesian_optimisation
    selected_routers = ["random", "cost_sensitive", "bayesian_optimisation"]

    for router_name in selected_routers:
        if router_name == "random":  # Handle random router separately
            router_results = []
            for i in range(args.random_iters):
                for threshold, accuracy, model_counts, total in benchmark.evaluate(
                    controller, ROUTER_CLS[router_name](), args.num_results, True
                ):
                    router_results.append(
                        {
                            "threshold": threshold,
                            "strong_percentage": model_counts[controller.model_pair.strong]
                            / total
                            * 100,
                            "accuracy": accuracy,
                        }
                    )
            router_results_df = (
                pd.DataFrame(router_results)
                .groupby(["strong_percentage"], as_index=False)
                .mean()
            )
            router_results_df["method"] = router_name
            all_results = pd.concat([all_results, router_results_df])
        elif router_name == "cost_sensitive":
            router_results = []

            # Instantiate the CostSensitiveRouter
            router_instance = ROUTER_CLS[router_name](
                strong_model_cost=1.5, weak_model_cost=1.0
            )
            print(f"Using router: {type(router_instance).__name__}")

            for threshold, accuracy, model_counts, total in benchmark.evaluate(
                controller, router_instance, args.num_results, args.overwrite_cache
            ):
                print(
                    f"Evaluating router: {type(router_instance).__name__} with threshold {threshold}..."
                )
                print(
                    f"Threshold: {threshold}, Accuracy: {accuracy}, Model Counts: {model_counts}"
                )

                strong_model_percentage = (
                    model_counts[controller.model_pair.strong] / total
                ) * 100
                weak_model_percentage = (
                    model_counts[controller.model_pair.weak] / total
                ) * 100

                result = {
                    "method": router_name,
                    "threshold": threshold,
                    "strong_percentage": strong_model_percentage,
                    "weak_percentage": weak_model_percentage,
                    "accuracy": accuracy,
                }
                router_results.append(result)
            router_results_df = pd.DataFrame(router_results)
            all_results = pd.concat([all_results, router_results_df])
        elif router_name == "bayesian_optimisation":
            router_results = []

            # Load best parameters for BayesianOptimisationRouter
            with open("best_params.json", "r") as f:
                best_params = json.load(f)

            print("Loaded best parameters for BayesianOptimisationRouter:", best_params)

            bayes_router = BayesianOptimisationRouter(
                low_threshold=best_params["low_threshold"],
                high_threshold=best_params["high_threshold"],
                strong_model_cost=best_params["strong_model_cost"],
                weak_model_cost=best_params["weak_model_cost"],
                complexity_scaling=best_params["complexity_scaling"],
            )

            print("Evaluating BayesianOptimisationRouter...")
            for threshold, accuracy, model_counts, total in benchmark.evaluate(
                controller, bayes_router, args.num_results, args.overwrite_cache
            ):
                print(
                    f"Threshold: {threshold}, Accuracy: {accuracy}, Model Counts: {model_counts}"
                )
                router_results.append(
                    {
                        "method": router_name,
                        "threshold": threshold,
                        "strong_percentage": model_counts[controller.model_pair.strong]
                        / total
                        * 100,
                        "weak_percentage": model_counts[controller.model_pair.weak]
                        / total
                        * 100,
                        "accuracy": accuracy,
                    }
                )
            router_results_df = pd.DataFrame(router_results)
            all_results = pd.concat([all_results, router_results_df])
    
    output_dir = "/content/RouteLLM/"

    # Define the output file name format
    output_file = os.path.join(output_dir, f"bo_{args.benchmark}.png")

    print(f"Saving results to {output_file}")

    # Call the generate_results function with the updated output file path
    generate_results(
        all_results,
        benchmark,
        args.benchmark,
        controller.model_pair,
        output_file,  
    )

