"""
Base framework for Sequential Decision Analytics and Modeling.

Abstract definitions of the Universal Modeling Framework (UMF) components:
- State: S^n
- Decision: x^n
- Exogenous Information: W^{n+1}
- Transition Function: S^{n+1} = S^M(S^n, x^n, W^{n+1})
- Objective Function: max_π E[Σ C(S^n, x^n, W^{n+1})]
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseState(ABC):
    """State S^n: all information needed to make a decision."""

    @abstractmethod
    def copy(self):
        """Return an independent copy of the state."""
        pass


class BaseSimulator(ABC):
    """Environment simulator: generates the 'truth' and noisy observations W^{n+1}."""

    @abstractmethod
    def observe(self, x):
        """Return W^{n+1} given decision x^n."""
        pass


class BasePolicy(ABC):
    """Policy X^π(S^n): maps state to decision."""

    @abstractmethod
    def decide(self, state):
        """Return decision x^n given state S^n."""
        pass


def evaluate_policy(policy_fn, state_factory, simulator_factory,
                    transition_fn, N_trials, K_simulations, base_seed=0):
    """
    Generic policy evaluation via Monte Carlo.

    F^π(S^0) ≈ (1/K) Σ_k Σ_n W^{n+1}_{x^n}(ω_k)

    Args:
        policy_fn: callable(state, rng) -> decision x^n
        state_factory: callable() -> initial state S^0
        simulator_factory: callable(rng) -> environment simulator
        transition_fn: callable(state, x, W) -> None (updates state in-place)
        N_trials: number of experiments per simulation
        K_simulations: number of Monte Carlo simulations
        base_seed: base seed for reproducibility

    Returns:
        dict with performance statistics
    """
    total_rewards = []
    cumulative_by_step = np.zeros(N_trials)

    for k in range(K_simulations):
        rng = np.random.RandomState(base_seed + k)
        sim = simulator_factory(rng)
        state = state_factory()

        total_reward = 0.0
        for n in range(N_trials):
            x_n = policy_fn(state, rng=rng)
            W = sim.observe(x_n)
            total_reward += W
            transition_fn(state, x_n, W)
            cumulative_by_step[n] += total_reward

        total_rewards.append(total_reward)

    return {
        'mean_total_reward': np.mean(total_rewards),
        'std_total_reward': np.std(total_rewards),
        'cumulative_by_step': cumulative_by_step / K_simulations,
    }
