# bayesian_optimisation.py

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Define the search space for optimisation
search_space = [
    Real(0.1, 1.0, name="low_threshold"),  # Dynamic threshold lower bound
    Real(0.2, 2.0, name="high_threshold"), # Dynamic threshold upper bound
# This use case involves scaling the threshold dynamically or applying transformations (e.g., multiplying by cost factors), hence a range beyond [0,1] is used. Thresholds adjusted dynamically by scaling factors, not used directly as a probability. 
    Real(1.0, 3.0, name="strong_model_cost"),  # Cost of the strong model
    Real(0.5, 2.0, name="weak_model_cost"),    # Cost of the weak model
    Real(0.1, 1.0, name="complexity_scaling"), # Scaling for complexity scores
]

print("Search space defined.")
