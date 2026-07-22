"""Pluggable sources of exogenous information ``W_{t+1}``.

The book (section 1.8.3) lists three ways to generate the sequence
``W_1(ω), ..., W_T(ω)`` when testing a policy:

1. sample from a **mathematical model** — :class:`ModelSource` (the default;
   wraps ``Model.exogenous_info``);
2. replay **historical data** / a dataset — :class:`DatasetSource`;
3. observe the **real world as it happens** (API/feed) — :class:`CallableSource`.

The engine accepts any :class:`ExogenousSource`, so switching from simulated
uncertainty to a real dataset (or a live feed) never touches the ``Model`` or
the ``Policy``::

    engine = sd.Engine(model, policy, horizon=24)                      # simulated
    engine = sd.Engine(model, policy, horizon=24,
                       exogenous_source=sd.DatasetSource(price_paths))  # dataset
    engine = sd.Engine(model, policy, horizon=24,
                       exogenous_source=sd.CallableSource(fetch_tick))  # live feed
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

import numpy as np

from .exceptions import ExogenousSourceError

__all__ = ["ExogenousSource", "ModelSource", "DatasetSource", "CallableSource"]


class ExogenousSource(ABC):
    """Where ``W_{t+1}`` comes from during a run.

    Subclass and implement :meth:`generate`; override :meth:`reset` if the
    source needs per-episode setup (e.g. selecting which historical sample
    path to replay).
    """

    def reset(self, episode: int, rng: np.random.Generator) -> None:
        """Called by the engine at the start of each episode."""

    @abstractmethod
    def generate(self, t: int, state: Any, decision: Any, rng: np.random.Generator) -> Any:
        """Return ``W_{t+1}`` for step ``t``."""


class ModelSource(ExogenousSource):
    """Default source: samples from ``model.exogenous_info`` (strategy 1)."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def generate(self, t: int, state: Any, decision: Any, rng: np.random.Generator) -> Any:
        return self.model.exogenous_info(state, decision, rng)


class DatasetSource(ExogenousSource):
    """Replay pre-recorded sample paths (strategy 2).

    Args:
        data: the recorded outcomes. Accepted layouts:
            * sequence of episodes, each a sequence of per-step outcomes
              (e.g. ``ndarray`` of shape ``(K, T)`` or ``(K, T, d)``, or a
              list of lists);
            * a single path of length ``T`` (replayed in every episode when
              ``single_path=True`` is implied by its shape — pass a 1-level
              sequence).
        builder: optional callable ``raw -> W`` converting a raw row (e.g. a
            numpy row or dict) into your ``ExogenousInfo`` object.
        cycle: if True, episode indices wrap around when the run asks for more
            episodes than the dataset contains; if False (default), that is an
            error — silent reuse of data can invalidate an experiment.
    """

    def __init__(
        self,
        data: Sequence[Any],
        builder: Callable[[Any], Any] | None = None,
        cycle: bool = False,
    ) -> None:
        self.data = data
        self.builder = builder
        self.cycle = cycle
        self._path: Sequence[Any] | None = None
        arr = np.asarray(data, dtype=object) if not isinstance(data, np.ndarray) else data
        self._is_single_path = getattr(arr, "ndim", 2) == 1 and not any(
            isinstance(el, (list, tuple, np.ndarray)) for el in list(data)[:1]
        )

    def n_episodes(self) -> int | None:
        """How many distinct episodes the dataset supports (None = unlimited)."""
        return None if self._is_single_path or self.cycle else len(self.data)

    def reset(self, episode: int, rng: np.random.Generator) -> None:
        if self._is_single_path:
            self._path = self.data
            return
        k = len(self.data)
        if episode >= k and not self.cycle:
            raise ExogenousSourceError(
                f"DatasetSource has {k} sample paths but episode {episode} was requested. "
                f"Run fewer episodes, provide more data, or pass cycle=True to reuse paths."
            )
        self._path = self.data[episode % k]

    def generate(self, t: int, state: Any, decision: Any, rng: np.random.Generator) -> Any:
        assert self._path is not None, "reset() must run before generate()"
        if t >= len(self._path):
            raise ExogenousSourceError(
                f"Sample path has {len(self._path)} steps but step t={t} was requested. "
                f"Reduce the engine horizon or provide longer paths."
            )
        raw = self._path[t]
        return self.builder(raw) if self.builder is not None else raw


class CallableSource(ExogenousSource):
    """Adapter for data arriving from the outside world (strategy 3).

    Wraps any callable ``fn(t, state, decision) -> W`` — an API poll, a message
    queue read, a sensor... The callable is invoked once per step; raise inside
    it (or return ``None`` and let the model fail) to abort a run.

    Example::

        def fetch_price_change(t, state, decision):
            return PriceChange(change=api.get_tick()["delta"])

        source = sd.CallableSource(fetch_price_change)
    """

    def __init__(self, fn: Callable[[int, Any, Any], Any]) -> None:
        if not callable(fn):
            raise TypeError("CallableSource expects a callable fn(t, state, decision) -> W.")
        self.fn = fn

    def generate(self, t: int, state: Any, decision: Any, rng: np.random.Generator) -> Any:
        return self.fn(t, state, decision)
