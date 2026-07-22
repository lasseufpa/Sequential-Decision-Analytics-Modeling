"""Well-known concrete policies.

Belief-based policies (Greedy/UCB/IE/Thompson) follow a naming convention for
the state fields they read — declare these on your ``State``:

* ``mu``: array of current estimates, one per alternative;
* ``sigma``: array of current uncertainties (std) — IE/Thompson;
* ``counts``: array with how many times each alternative was tried — UCB;
* ``n``: total number of observations so far — UCB.

The requirement is checked before the run starts (see
``Policy.state_requirements``), so a mismatched state fails fast and loudly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._random import RandomState, check_random_state
from .base import Policy

__all__ = [
    "ThresholdPolicy",
    "GreedyPolicy",
    "UCBPolicy",
    "IEPolicy",
    "ThompsonSamplingPolicy",
]


class ThresholdPolicy(Policy):
    """Order-up-to rule (book eq. 1.7): if the resource drops below
    ``theta_min``, replenish up to ``theta_max``; otherwise do nothing.

    A classic **PFA** — a closed-form function of the state, no optimization.
    """

    state_requirements = ("resource",)

    def __init__(self, theta_min: float, theta_max: float) -> None:
        if theta_max < theta_min:
            raise ValueError(f"theta_max ({theta_max}) must be >= theta_min ({theta_min}).")
        self.theta_min = theta_min
        self.theta_max = theta_max

    def decide(self, state: Any, model: Any = None) -> float:
        if state.resource < self.theta_min:
            return self.theta_max - state.resource
        return 0.0

    def tags(self) -> dict[str, Any]:
        return {"policy_class": "PFA", "needs_model": False}


class GreedyPolicy(Policy):
    """Pure exploitation: ``argmax_x mu_x``. No exploration, no parameters."""

    state_requirements = ("mu",)

    def decide(self, state: Any, model: Any = None) -> int:
        return int(np.argmax(state.mu))

    def tags(self) -> dict[str, Any]:
        return {"policy_class": "PFA", "needs_model": False}


class UCBPolicy(Policy):
    """Upper confidence bound: ``argmax_x [ mu_x + theta * sqrt(log n / counts_x) ]``.

    Untried alternatives get infinite priority. A **CFA** — an argmax over a
    parametrically modified objective, with ``theta`` tuned by simulation.
    """

    state_requirements = ("mu", "counts", "n")

    def __init__(self, theta: float = 1.0) -> None:
        self.theta = theta

    def decide(self, state: Any, model: Any = None) -> int:
        mu = np.asarray(state.mu, dtype=float)
        counts = np.asarray(state.counts, dtype=float)
        n = max(int(state.n), 1)
        scores = np.full(mu.shape, np.inf)
        tried = counts > 0
        scores[tried] = mu[tried] + self.theta * np.sqrt(np.log(n) / counts[tried])
        return int(np.argmax(scores))

    def tags(self) -> dict[str, Any]:
        return {"policy_class": "CFA", "needs_model": False}


class IEPolicy(Policy):
    """Interval estimation: ``argmax_x [ mu_x + theta * sigma_x ]``. A **CFA**."""

    state_requirements = ("mu", "sigma")

    def __init__(self, theta: float = 1.0) -> None:
        self.theta = theta

    def decide(self, state: Any, model: Any = None) -> int:
        return int(np.argmax(np.asarray(state.mu) + self.theta * np.asarray(state.sigma)))

    def tags(self) -> dict[str, Any]:
        return {"policy_class": "CFA", "needs_model": False}


class ThompsonSamplingPolicy(Policy):
    """Thompson sampling: draw one scenario per alternative from the current
    belief ``N(mu_x, (theta*sigma_x)^2)`` and pick the best draw. Exploration
    emerges from the sampling itself.
    """

    state_requirements = ("mu", "sigma")

    def __init__(self, theta: float = 1.0, random_state: RandomState = None) -> None:
        self.theta = theta
        self.random_state = random_state
        self._rng = check_random_state(random_state)

    def decide(self, state: Any, model: Any = None) -> int:
        samples = self._rng.normal(np.asarray(state.mu), self.theta * np.asarray(state.sigma))
        return int(np.argmax(samples))

    def tags(self) -> dict[str, Any]:
        return {"policy_class": "CFA", "needs_model": False, "stochastic": True}
