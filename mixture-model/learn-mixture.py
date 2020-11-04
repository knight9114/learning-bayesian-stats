# ---------------------------------------------------------------
#   Learn Coin-Flip Mixture
# ---------------------------------------------------------------

# General Imports
import argparse


# Special Imports
import numpy as np
import pymc3 as pm
from matplotlib import pyplot as plt


# ---------------------------------------------------------------
#   Main
# ---------------------------------------------------------------
def main():
    # Hyperparameters
    n_flips = 125
    n_coins = 10
    n_draws = 5000
    n_init_steps = 10000
    n_burn_in_steps = 1000

    # Create Causal Distribution
    causal_probs = np.random.uniform(size=n_coins)

    # Create Observations
    X = np.array([np.random.choice(2, p=[1 - p_, p_], size=n_flips)
        for i, p_ in enumerate(causal_probs)]).T

    # Create Model
    with pm.Model() as model:
        ps = pm.Beta('probs', alpha=1, beta=1, shape=n_coins)
        components = pm.Bernoulli.dist(p=ps, shape=n_coins)
        w = pm.Dirichlet('w', a=np.ones(n_coins))
        mix = pm.Mixture('mix', w=w, comp_dists=components, observed=X)

    # Train Model
    with model:
        trace = pm.sample(n_draws, n_init=n_init_steps, tune=n_burn_in_steps)

    # Display Results
    pm.plot_trace(trace, var_names=['w', 'probs'])
    plt.show()
    pm.plot_posterior(trace, var_names=['w', 'probs'])
    plt.show()


# ---------------------------------------------------------------
#   Argument Parsing
# ---------------------------------------------------------------
def parse_input():
    """
    """
    # Root Parser
    return None


# ---------------------------------------------------------------
#   Script Mode
# ---------------------------------------------------------------
if __name__ == '__main__':
    main()
