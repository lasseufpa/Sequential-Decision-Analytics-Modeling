"""Rich result object returned by ``Engine.run``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .history import History
    from .metrics import MetricSet

__all__ = ["RunResult"]


class RunResult:
    """Aggregated outcome of a run: metrics + full step history.

    Common accessors are properties (``mean_reward``, ``std_reward``,
    ``cumulative_by_step``, decision timing); anything else is reachable via
    ``result.metrics[<name>]`` (custom metrics included) and ``to_frame()``.
    """

    def __init__(
        self,
        history: "History",
        metrics: "MetricSet",
        episodes: int,
        horizon: int,
        policy_name: str,
        model_name: str,
    ) -> None:
        self.history = history
        self.metrics = metrics
        self.episodes = episodes
        self.horizon = horizon
        self.policy_name = policy_name
        self.model_name = model_name
        # snapshot core results so the engine can be re-run without mutating us
        self._reward = metrics["episode_reward"]
        self._cumulative = np.array(metrics["cumulative_by_step"], copy=True)
        self._timing = metrics["decision_time"]

    # ------------------------------------------------------------------ #

    @property
    def per_episode_reward(self) -> np.ndarray:
        """Total reward of each episode, shape ``(episodes,)``."""
        return self._reward["per_episode"]

    @property
    def mean_reward(self) -> float:
        """``F^π(S_0)``: mean total reward across sample paths."""
        return self._reward["mean"]

    @property
    def std_reward(self) -> float:
        return self._reward["std"]

    def quantiles(self, qs: Sequence[float]) -> np.ndarray:
        """Reward quantiles across episodes (worst/best case views)."""
        return np.quantile(self.per_episode_reward, qs)

    @property
    def cumulative_by_step(self) -> np.ndarray:
        """Mean cumulative reward per time step, shape ``(horizon,)``."""
        return self._cumulative

    @property
    def mean_decision_time_ms(self) -> float:
        return self._timing["mean_ms"]

    @property
    def max_decision_time_ms(self) -> float:
        return self._timing["max_ms"]

    # ------------------------------------------------------------------ #

    def to_frame(self, include_next_state: bool = False):
        """Raw step records as a tidy ``pandas.DataFrame``."""
        return self.history.to_frame(include_next_state=include_next_state)

    def plot_cumulative(self, ax: Any = None, label: str | None = None):
        """Plot the mean cumulative reward curve; returns the matplotlib axis."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "plot_cumulative requires matplotlib: pip install 'pysdm[analysis]'."
            ) from err
        if ax is None:
            _, ax = plt.subplots()
        steps = np.arange(1, len(self.cumulative_by_step) + 1)
        ax.plot(steps, self.cumulative_by_step, label=label or self.policy_name)
        ax.set_xlabel("t")
        ax.set_ylabel("mean cumulative reward")
        ax.grid(alpha=0.3)
        if label or self.policy_name:
            ax.legend()
        return ax

    def summary(self) -> dict[str, Any]:
        """All metric results plus run identification, as a plain dict."""
        out: dict[str, Any] = {
            "model": self.model_name,
            "policy": self.policy_name,
            "episodes": self.episodes,
            "horizon": self.horizon,
        }
        out.update(self.metrics.results())
        return out

    def __repr__(self) -> str:
        return (
            f"RunResult({self.model_name} × {self.policy_name}, "
            f"episodes={self.episodes}, horizon={self.horizon}, "
            f"mean_reward={self.mean_reward:.4f} ± {self.std_reward:.4f})"
        )
