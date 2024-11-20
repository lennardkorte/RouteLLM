import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import yaml
from routellm.controller import Controller
from routellm.evals.benchmarks import GSM8K, MMLU, MTBench

# Lazy import to avoid circular dependencies
def load_benchmark(benchmark_name, controller, overwrite_cache):
    
    if benchmark_name == "gsm8k":
        return GSM8K(controller.model_pair, overwrite_cache)
    elif benchmark_name == "mmlu":
        return MMLU(domains=["STEM", "Humanities"], routed_pair=controller.model_pair, overwrite_cache=overwrite_cache)
    elif benchmark_name == "mt-bench":
        return MTBench(controller.model_pair, overwrite_cache)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


class BayesianOptimisation:
    def __init__(self, benchmark_name, config_path="config.example.yaml"):
        self.benchmark_name = benchmark_name

        # Load config
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        # Initialize Controller
        self.controller = Controller(
            routers=[],  # Placeholder, dynamically instantiated
            config=self.config,
            strong_model="gpt-4-1106-preview",
            weak_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            progress_bar=False,
        )

        # Define the search space for optimization
        self.search_space = [
            Real(0.1, 1.0, name="low_threshold"),          # Lower bound of dynamic threshold
            Real(0.2, 2.0, name="high_threshold"),         # Upper bound of dynamic threshold
            Real(1.0, 3.0, name="strong_model_cost"),      # Cost of strong model
            Real(0.5, 2.0, name="weak_model_cost"),        # Cost of weak model
            Real(0.1, 2.0, name="complexity_scaling"),     # Scaling for prompt complexity
        ]

        print("Search space defined.")

    def objective(self, params):
        """Objective function for Bayesian Optimization."""
        # Extract parameters
        low_threshold = params["low_threshold"]
        high_threshold = params["high_threshold"]
        strong_model_cost = params["strong_model_cost"]
        weak_model_cost = params["weak_model_cost"]
        complexity_scaling = params["complexity_scaling"]

        # Dynamically instantiate router
        from routellm.routers.routers import BayesianOptimisationRouter
        router_instance = BayesianOptimisationRouter(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            strong_model_cost=strong_model_cost,
            weak_model_cost=weak_model_cost,
            complexity_scaling=complexity_scaling,
        )

        # Load benchmark
        benchmark = load_benchmark(self.benchmark_name, self.controller, overwrite_cache=True)

        # Evaluate router on the benchmark
        results = []
        for threshold, accuracy, model_counts, total in benchmark.evaluate(
            self.controller, router_instance, num_results=10, overwrite_router_cache=True
        ):
            results.append(accuracy)

        # Return negative average accuracy (for minimization)
        avg_accuracy = sum(results) / len(results)
        print(f"Average accuracy for params {params}: {avg_accuracy:.4f}")
        return -avg_accuracy

    def run_optimization(self, n_calls=50):
        """Run Bayesian Optimization."""
        from skopt.utils import point_asdict

        # Perform optimization
        result = gp_minimize(
            func=lambda x: self.objective(point_asdict({dim.name: dim for dim in self.search_space}, x)),
            dimensions=self.search_space,
            n_calls=n_calls,
            random_state=0,
        )

        print("Optimization completed.")
        print("Best parameters:", point_asdict({dim.name: dim for dim in self.search_space}, result.x))
        print("Best score:", -result.fun)
        return result


if __name__ == "__main__":
    # Example usage
    bayes_opt = BayesianOptimisation(benchmark_name="mt-bench")
    opt_result = bayes_opt.run_optimization(n_calls=20)


