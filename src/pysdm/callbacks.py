"""Callbacks: observe a run while it happens (PyTorch/Keras style).

Attach any number of callbacks to the engine to log, stream metrics to a
dashboard, checkpoint a learning policy, or stop early::

    engine = sd.Engine(model, policy, horizon=30,
                       callbacks=[sd.ProgressLogger(every=200)])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .history import StepRecord

if TYPE_CHECKING:  # pragma: no cover
    from .engine import Engine

__all__ = ["Callback", "ProgressLogger"]


class Callback:
    """Base class; override any subset of the hooks (all default to no-op)."""

    def on_run_begin(self, engine: "Engine", episodes: int) -> None: ...

    def on_episode_begin(self, engine: "Engine", episode: int) -> None: ...

    def on_step(self, engine: "Engine", record: StepRecord) -> None: ...

    def on_episode_end(self, engine: "Engine", episode: int) -> None: ...

    def on_run_end(self, engine: "Engine", result: Any) -> None: ...


class ProgressLogger(Callback):
    """Print running metrics every ``every`` episodes — online view of the run.

    Args:
        every: logging period, in episodes.
        sink: where lines go (default ``print``); pass e.g. ``logger.info``.
    """

    def __init__(self, every: int = 100, sink: Callable[[str], None] = print) -> None:
        if every < 1:
            raise ValueError("every must be >= 1.")
        self.every = every
        self.sink = sink

    def on_run_begin(self, engine: "Engine", episodes: int) -> None:
        self._episodes = episodes

    def on_episode_end(self, engine: "Engine", episode: int) -> None:
        done = episode + 1
        if done % self.every and done != self._episodes:
            return
        reward = engine.metrics["episode_reward"]
        timing = engine.metrics["decision_time"]
        self.sink(
            f"[pysdm] episode {done}/{self._episodes}  "
            f"mean_reward={reward['mean']:.4f} ± {reward['std']:.4f}  "
            f"decide={timing['mean_ms']:.3f}ms"
        )
