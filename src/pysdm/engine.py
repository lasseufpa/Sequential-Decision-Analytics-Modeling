"""The Engine: prepares everything, then runs the decision loop.

One episode is one sample path ω, stepped over time ``t = 0..horizon-1``::

    x_t   = policy.decide(S_t)              # decision
    W_t+1 = exogenous_source.generate(...)  # information (model, dataset or feed)
    C_t   = model.objective(S_t, x_t, W_t+1)
    S_t+1 = model.transition(S_t, x_t, W_t+1)
    policy.update(...)                      # learning hook

``run(episodes=K)`` repeats it over K sample paths, feeding metrics and
callbacks online, and returns a :class:`~pysdm.result.RunResult`.

Fair comparison (*common random numbers*): a single ``random_state`` derives
one independent stream per episode, so two engines with the same seed and
different policies face exactly the same uncertainty.
"""

from __future__ import annotations

import inspect
import time
import warnings
from typing import Any, Iterable, Sequence

from ._random import RandomState, spawn_generators
from .callbacks import Callback
from .exceptions import ModelDefinitionError
from .exogenous import ExogenousSource, ModelSource
from .history import History, StepRecord
from .metrics import Metric, MetricSet, default_metrics
from .result import RunResult

__all__ = ["Engine"]

_MODEL_METHODS = ("initial_state", "exogenous_info", "transition", "objective")


def _validate_model(model: Any) -> None:
    missing = [m for m in _MODEL_METHODS if not callable(getattr(model, m, None))]
    if missing:
        raise ModelDefinitionError(
            f"{type(model).__name__} is not a valid model: missing callable "
            f"method(s) {', '.join(missing)}. Subclass sd.Model and implement them."
        )
    if isinstance(model, type):
        raise TypeError(
            f"Pass an *instance* of {model.__name__}, not the class itself "
            f"(did you forget the parentheses? Engine(model={model.__name__}(...), ...))."
        )


def _validate_policy(policy: Any) -> None:
    if isinstance(policy, type):
        raise TypeError(
            f"Pass an *instance* of {policy.__name__}, not the class itself."
        )
    if not callable(getattr(policy, "decide", None)):
        raise TypeError(
            f"{type(policy).__name__} is not a valid policy: it has no decide() "
            f"method. Subclass sd.Policy and implement decide(state, model=None)."
        )


def _decide_caller(policy: Any):
    """Build the fastest call matching the policy's decide() signature.

    Policies may declare ``decide(state)``, ``decide(state, model)`` or
    ``decide(state, model, t)`` — extra context is passed only if accepted,
    so simple policies stay simple.
    """
    try:
        params = inspect.signature(policy.decide).parameters
    except (TypeError, ValueError):
        return lambda state, model, t: policy.decide(state, model)
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    takes_model = "model" in params or has_kwargs
    takes_t = "t" in params or has_kwargs
    if takes_model and takes_t:
        return lambda state, model, t: policy.decide(state, model=model, t=t)
    if takes_model:
        return lambda state, model, t: policy.decide(state, model=model)
    if takes_t:
        return lambda state, model, t: policy.decide(state, t=t)
    return lambda state, model, t: policy.decide(state)


class Engine:
    """Runs ``Model`` + ``Policy`` over a time horizon, many times.

    Args:
        model: the problem (a :class:`sd.Model` instance).
        policy: the decision strategy (a :class:`sd.Policy` instance).
        horizon: number of decision epochs ``T`` per episode (steps ``t=0..T-1``).
        random_state: seed/Generator for full reproducibility.
        exogenous_source: where ``W_{t+1}`` comes from. Default: the model's own
            ``exogenous_info`` (simulation); pass :class:`sd.DatasetSource` /
            :class:`sd.CallableSource` for real data.
        metrics: extra :class:`sd.Metric` instances (the built-ins —
            ``episode_reward``, ``cumulative_by_step``, ``decision_time`` —
            are always attached).
        callbacks: :class:`sd.Callback` instances for online monitoring.
        record_history: keep per-step records (disable for huge runs).
    """

    def __init__(
        self,
        model: Any,
        policy: Any,
        horizon: int,
        random_state: RandomState = None,
        exogenous_source: ExogenousSource | None = None,
        metrics: Iterable[Metric] = (),
        callbacks: Sequence[Callback] = (),
        record_history: bool = True,
    ) -> None:
        _validate_model(model)
        _validate_policy(policy)
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError(f"horizon must be a positive int, got {horizon!r}.")
        if exogenous_source is not None and not isinstance(exogenous_source, ExogenousSource):
            raise TypeError(
                f"exogenous_source must be an ExogenousSource, got "
                f"{type(exogenous_source).__name__!r}. Use sd.DatasetSource / "
                f"sd.CallableSource, or subclass sd.ExogenousSource."
            )
        self.model = model
        self.policy = policy
        self.horizon = horizon
        self.random_state = random_state
        self.exogenous_source = exogenous_source or ModelSource(model)
        self.callbacks: list[Callback] = list(callbacks)
        self.record_history = record_history
        self.metrics = MetricSet(default_metrics())
        for metric in metrics:
            self.metrics.add(metric)
        self.history = History()
        self._mutation_warned = False

    def add_metric(self, metric: Metric, name: str | None = None) -> "Engine":
        """Attach a custom metric (see :class:`sd.Metric`). Returns ``self``."""
        self.metrics.add(metric, name=name)
        return self

    def add_callback(self, callback: Callback) -> "Engine":
        self.callbacks.append(callback)
        return self

    # ------------------------------------------------------------------ #

    def _check_first_step(self, contribution: Any, next_state: Any, state: Any) -> None:
        """Loud, early feedback on the most common modeling mistakes."""
        try:
            float(contribution)
        except (TypeError, ValueError):
            raise ModelDefinitionError(
                f"{type(self.model).__name__}.objective must return a number, got "
                f"{type(contribution).__name__!r}."
            ) from None
        if next_state is None:
            raise ModelDefinitionError(
                f"{type(self.model).__name__}.transition returned None — it must "
                f"return the next state S_{{t+1}} (did you forget the return?)."
            )
        if next_state is state and not self._mutation_warned:
            warnings.warn(
                f"{type(self.model).__name__}.transition returned the *same* state "
                f"object it received. Transitions should build a new state "
                f"(state.replace(...)); mutating in place breaks lookahead "
                f"policies and history records.",
                stacklevel=2,
            )
            self._mutation_warned = True

    def run(self, episodes: int = 1) -> RunResult:
        """Simulate ``episodes`` sample paths; returns a :class:`RunResult`.

        Metrics and history are reset at the start of each call, so repeated
        ``run``s are independent experiments with identical sample paths
        (same ``random_state``).
        """
        if not isinstance(episodes, int) or episodes < 1:
            raise ValueError(f"episodes must be a positive int, got {episodes!r}.")
        self.metrics.reset()
        self.history = History()
        decide = _decide_caller(self.policy)
        rngs = spawn_generators(self.random_state, episodes)

        # foolproofing before any work happens
        s0 = self.model.initial_state()
        if s0 is None:
            raise ModelDefinitionError(
                f"{type(self.model).__name__}.initial_state returned None."
            )
        if hasattr(self.policy, "validate_state"):
            self.policy.validate_state(s0)

        for cb in self.callbacks:
            cb.on_run_begin(self, episodes)

        first_step = True
        for k in range(episodes):
            rng = rngs[k]
            self.exogenous_source.reset(k, rng)
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            state = self.model.initial_state()
            for cb in self.callbacks:
                cb.on_episode_begin(self, k)

            for t in range(self.horizon):
                tic = time.perf_counter()
                decision = decide(state, self.model, t)
                decision_time = time.perf_counter() - tic

                exog = self.exogenous_source.generate(t, state, decision, rng)
                contribution = self.model.objective(state, decision, exog)
                next_state = self.model.transition(state, decision, exog)
                if first_step:
                    self._check_first_step(contribution, next_state, state)
                    first_step = False

                if hasattr(self.policy, "update"):
                    self.policy.update(t, state, decision, exog, next_state)

                record = StepRecord(
                    episode=k,
                    t=t,
                    state=state,
                    decision=decision,
                    exog_info=exog,
                    contribution=float(contribution),
                    next_state=next_state,
                    decision_time=decision_time,
                )
                self.metrics.update(record)
                if self.record_history:
                    self.history.append(record)
                for cb in self.callbacks:
                    cb.on_step(self, record)
                state = next_state

            self.metrics.on_episode_end(k)
            for cb in self.callbacks:
                cb.on_episode_end(self, k)

        result = RunResult(
            history=self.history,
            metrics=self.metrics,
            episodes=episodes,
            horizon=self.horizon,
            policy_name=type(self.policy).__name__,
            model_name=type(self.model).__name__,
        )
        for cb in self.callbacks:
            cb.on_run_end(self, result)
        return result
