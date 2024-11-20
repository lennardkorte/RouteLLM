import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml
from pandarallel import pandarallel

from routellm.controller import Controller
from routellm.evals.benchmarks import GSM8K, MMLU, MTBench
from routellm.evals.mmlu.domains import ALL_MMLU_DOMAINS
from routellm.routers.routers import CostSensitiveRouter, ROUTER_CLS
from routellm.routers.bayesian_optimisation import BayesianOptimisationRouter



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

    file_name = f"{output}/{benchmark_name}.png"
    print("Saving plot to", file_name)
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

    # Define the objective function for Bayesian Optimization
    def objective(threshold):
        router = BayesianOptimisationRouter(threshold=threshold[0], ...)
        results = benchmark.evaluate(controller, router, ...)
        accuracy = results["accuracy"]
        return -accuracy  # Negative because we want to maximize accuracy

    # Define the search space (e.g., thresholds between 0 and 1)
    search_space = [(0.0, 1.0)]

    # Perform Bayesian Optimization
    from skopt import gp_minimize
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=50,
        random_state=42
    )
    print("Best parameters:", result.x)
    print("Best performance:", -result.fun)


    all_results = pd.DataFrame()
    # Ensure reproducibility on a per-router basis
    random.seed(0)

    for router_name in controller.routers:
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
            router_results_df["method"] = router_name  # Use the router name directly
            all_results = pd.concat([all_results, router_results_df])
        else:
            router_results = []

            # Dynamically instantiate the router instance from ROUTER_CLS
            router_instance = ROUTER_CLS[router_name](
                strong_model_cost=1.5, weak_model_cost=1.0
            )
            print(f"Using router: {type(router_instance).__name__}")

            # Evaluate the benchmark with the instantiated router
            for threshold, accuracy, model_counts, total in benchmark.evaluate(
                controller, router_instance, args.num_results, args.overwrite_cache
            ):
                print(
                    f"Evaluating router: {type(router_instance).__name__} with threshold {threshold}..."
                )
                print(
                    f"Threshold: {threshold}, Accuracy: {accuracy}, Model Counts: {model_counts}"
                )

                # Calculate the percentage of prompts routed to each model
                strong_model_percentage = (model_counts[controller.model_pair.strong] / total) * 100
                weak_model_percentage = (model_counts[controller.model_pair.weak] / total) * 100

                print(
                    f"Strong Model Percentage: {strong_model_percentage:.2f}%, Weak Model Percentage: {weak_model_percentage:.2f}%"
                )

                # Collect results
                result = {
                    "method": router_name,
                    "threshold": threshold,
                    "strong_percentage": strong_model_percentage,
                    "weak_percentage": weak_model_percentage,
                    "accuracy": accuracy,
                }
                router_results.append(result)
            # Create a DataFrame from the router results
            router_results_df = pd.DataFrame(router_results)

            # Calculate overall averages for this router
            average_strong_percentage = router_results_df["strong_percentage"].mean()
            average_weak_percentage = router_results_df["weak_percentage"].mean()

            print(
                f"Overall Strong Model Usage for {type(router_instance).__name__}: {average_strong_percentage:.2f}%, "
                f"Overall Weak Model Usage for {type(router_instance).__name__}: {average_weak_percentage:.2f}%"
            )

            # Append the results to all_results DataFrame
            all_results = pd.concat([all_results, pd.DataFrame(router_results)])

    generate_results(
        all_results,
        benchmark,
        args.benchmark,
        controller.model_pair,
        args.output,
    )

