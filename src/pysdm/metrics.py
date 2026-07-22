"""Extensible metrics, computed online while the engine runs.

A :class:`Metric` sees every :class:`~pysdm.history.StepRecord` as it happens
(``update``) and can be queried at any time (``result``) — during the run
(live monitoring via callbacks) or after it. To create a custom measure,
subclass and register it on the engine::

    class FractionIdle(sd.Metric):
        name = "fraction_idle"
        def reset(self):
            self.idle = 0
            self.total = 0
        def update(self, record):
            self.total += 1
            self.idle += record.decision == 0
        def result(self):
            return self.idle / max(self.total, 1)

    engine.add_metric(FractionIdle())
    engine.run(episodes=500).metrics["fraction_idle"]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .history import StepRecord

__all__ = ["Metric", "MetricSet", "EpisodeReward", "CumulativeByStep", "DecisionTime"]


class Metric(ABC):
    """Streaming measure over a run. Lifecycle: ``reset`` -> ``update``* -> ``result``."""

    #: Registry key; defaults to the class name in snake_case.
    name: str | None = None

    def reset(self) -> None:
        """Clear accumulated data (called once per ``engine.run``)."""

    @abstractmethod
    def update(self, record: StepRecord) -> None:
        """Ingest one step."""

    def on_episode_end(self, episode: int) -> None:
        """Optional hook after each episode."""

    @abstractmethod
    def result(self) -> Any:
        """Current value. Must be callable mid-run (online monitoring)."""

    def key(self) -> str:
        if self.name:
            return self.name
        cls = type(self).__name__
        return "".join(("_" + c.lower() if c.isupper() else c) for c in cls).lstrip("_")


class MetricSet(Mapping[str, Any]):
    """Named collection of metrics attached to an engine.

    Mapping access returns the metric **result** (``engine.metrics["episode_reward"]``);
    use :meth:`get_metric` for the metric object itself.
    """

    def __init__(self, metrics: Iterable[Metric] = ()) -> None:
        self._metrics: dict[str, Metric] = {}
        for m in metrics:
            self.add(m)

    def add(self, metric: Metric, name: str | None = None) -> None:
        if not isinstance(metric, Metric):
            raise TypeError(
                f"Expected a Metric subclass instance, got {type(metric).__name__!r}. "
                f"Extend sd.Metric and implement update()/result()."
            )
        key = name or metric.key()
        if key in self._metrics:
            raise ValueError(f"A metric named {key!r} already exists.")
        metric.reset()
        self._metrics[key] = metric

    def get_metric(self, name: str) -> Metric:
        return self._metrics[name]

    def reset(self) -> None:
        for m in self._metrics.values():
            m.reset()

    def update(self, record: StepRecord) -> None:
        for m in self._metrics.values():
            m.update(record)

    def on_episode_end(self, episode: int) -> None:
        for m in self._metrics.values():
            m.on_episode_end(episode)

    def results(self) -> dict[str, Any]:
        return {name: m.result() for name, m in self._metrics.items()}

    # Mapping protocol -> results, so `engine.metrics["x"]` reads naturally
    def __getitem__(self, name: str) -> Any:
        return self._metrics[name].result()

    def __iter__(self) -> Iterator[str]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        return f"MetricSet({list(self._metrics)})"


# ---------------------------------------------------------------------- #
# Built-in metrics (always attached by default)                          #
# ---------------------------------------------------------------------- #


class EpisodeReward(Metric):
    """Total contribution per episode; result gives mean/std/per-episode array."""

    name = "episode_reward"

    def reset(self) -> None:
        self._totals: list[float] = []
        self._current = 0.0
        self._open = False

    def update(self, record: StepRecord) -> None:
        self._current += record.contribution
        self._open = True

    def on_episode_end(self, episode: int) -> None:
        if self._open:
            self._totals.append(self._current)
            self._current = 0.0
            self._open = False

    def result(self) -> dict[str, Any]:
        totals = np.asarray(self._totals, dtype=float)
        if totals.size == 0:
            return {"mean": np.nan, "std": np.nan, "per_episode": totals}
        return {"mean": float(totals.mean()), "std": float(totals.std()), "per_episode": totals}


class CumulativeByStep(Metric):
    """Mean cumulative reward curve over time (averaged across episodes)."""

    name = "cumulative_by_step"

    def reset(self) -> None:
        self._sums: list[float] = []
        self._episodes = 0
        self._cursor = 0
        self._running = 0.0

    def update(self, record: StepRecord) -> None:
        if record.t == 0:
            self._running = 0.0
            self._cursor = 0
        self._running += record.contribution
        if self._cursor >= len(self._sums):
            self._sums.append(0.0)
        self._sums[self._cursor] += self._running
        self._cursor += 1

    def on_episode_end(self, episode: int) -> None:
        self._episodes += 1

    def result(self) -> np.ndarray:
        if self._episodes == 0:
            return np.asarray(self._sums, dtype=float)
        return np.asarray(self._sums, dtype=float) / self._episodes


class DecisionTime(Metric):
    """Computational cost of ``policy.decide`` — mean and worst case, in ms."""

    name = "decision_time"

    def reset(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._max = 0.0

    def update(self, record: StepRecord) -> None:
        self._count += 1
        self._sum += record.decision_time
        self._max = max(self._max, record.decision_time)

    def result(self) -> dict[str, float]:
        mean = self._sum / self._count if self._count else float("nan")
        return {"mean_ms": mean * 1e3, "max_ms": self._max * 1e3}


def default_metrics() -> list[Metric]:
    return [EpisodeReward(), CumulativeByStep(), DecisionTime()]
