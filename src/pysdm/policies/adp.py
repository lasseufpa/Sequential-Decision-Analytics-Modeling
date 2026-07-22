"""Forward approximate dynamic programming (book, section 8.6.4, figure 8.7).

The policy decides with

    x_t = argmax_x [ C_t(S_t, x) + V(S^x_t) ]

where ``V`` is a *learned* approximation of the value of the post-decision
state ``S^x_t`` (the state right after deciding, before new information
arrives). Training simulates forward in time — it never enumerates the state
space, which is why it scales to complex states.

The model must implement two optional hooks:

* ``decision_space(state)`` — the feasible decisions to argmax over;
* ``post_decision_state(state, decision)`` — ``S^x_t``.

Example (energy storage, ``S_t = (R_t, p_t)``)::

    def post_decision_state(self, state, decision):
        return state.replace(resource=state.resource + decision)
"""

from __future__ import annotations

from typing import Any, Callable, Hashable

import numpy as np

from .._random import RandomState, check_random_state
from ..exceptions import NotFittedError
from .base import Policy

__all__ = ["TabularValueFunction", "ForwardADPPolicy"]


def _default_key(post_state: Any) -> Hashable:
    """Turn a post-decision state into a dict key.

    Works out of the box for hashable scalars/tuples and for declarative
    elements with hashable fields. Continuous states should be discretized
    with an explicit ``key`` function (e.g. rounding).
    """
    to_dict = getattr(post_state, "to_dict", None)
    value = tuple(to_dict().values()) if callable(to_dict) else post_state
    try:
        hash(value)
    except TypeError as err:
        raise TypeError(
            f"Post-decision state {post_state!r} is not hashable. Pass an explicit "
            f"key function to ForwardADPPolicy, e.g. "
            f"key=lambda s: (round(s.resource, 1), round(s.price, 1))."
        ) from err
    return value


class TabularValueFunction:
    """Lookup-table approximation ``V_t(S^x_t)`` with per-entry stepsizes.

    Args:
        key: maps a post-decision state to a hashable table key (use it to
            discretize continuous states). Default handles hashable states.
        initial_value: optimistic initial values encourage exploration.
    """

    def __init__(
        self,
        key: Callable[[Any], Hashable] | None = None,
        initial_value: float = 0.0,
    ) -> None:
        self.key = key or _default_key
        self.initial_value = initial_value
        self._table: dict[Hashable, float] = {}
        self._visits: dict[Hashable, int] = {}

    def _index(self, t: int | None, post_state: Any) -> Hashable:
        return (t, self.key(post_state))

    def predict(self, t: int | None, post_state: Any) -> float:
        """Current estimate of the value of ``post_state`` at time ``t``."""
        return self._table.get(self._index(t, post_state), self.initial_value)

    def update(self, t: int | None, post_state: Any, target: float, alpha: float | None = None) -> None:
        """Smooth the stored value toward ``target``: ``V <- (1-a)V + a*target``.

        With ``alpha=None`` a harmonic stepsize ``1/m`` is used, where ``m`` is
        the number of visits to this entry (the book's typical choice).
        """
        idx = self._index(t, post_state)
        self._visits[idx] = self._visits.get(idx, 0) + 1
        a = 1.0 / self._visits[idx] if alpha is None else alpha
        current = self._table.get(idx, self.initial_value)
        self._table[idx] = (1.0 - a) * current + a * target

    def __len__(self) -> int:
        return len(self._table)


class ForwardADPPolicy(Policy):
    """Value-function policy trained by forward ADP (figure 8.7).

    Usage::

        policy = sd.ForwardADPPolicy(key=lambda s: (s.resource, round(s.price)))
        policy.fit(model, horizon=24, n_iterations=50, n_samples=20, random_state=0)
        engine = sd.Engine(model, policy, horizon=24, random_state=1)

    Args:
        value_function: a fitted/blank :class:`TabularValueFunction` (or any
            object with ``predict``/``update``). Default: a fresh table.
        key: shortcut to build the default table with a discretization key.
        exploration: epsilon-greedy exploration rate *during training only*
            (figure 8.7 uses pure exploitation; a little exploration helps
            when initial values are not optimistic).
        time_indexed: if True (default) the value function is indexed by
            ``t`` (finite-horizon); if False a single table is shared across
            time (stationary problems).
    """

    def __init__(
        self,
        value_function: TabularValueFunction | None = None,
        key: Callable[[Any], Hashable] | None = None,
        exploration: float = 0.0,
        time_indexed: bool = True,
        random_state: RandomState = None,
    ) -> None:
        self.value_function = value_function
        self.key = key
        self.exploration = exploration
        self.time_indexed = time_indexed
        self.random_state = random_state
        self._vf: TabularValueFunction | None = value_function
        self._fitted = value_function is not None

    # ------------------------------------------------------------------ #

    def _vf_time(self, t: int | None) -> int | None:
        return t if self.time_indexed else None

    def _argmax(self, state: Any, model: Any, t: int | None) -> Any:
        assert self._vf is not None
        best_x, best_value = None, -np.inf
        for x in model.decision_space(state):
            value = model.contribution(state, x) + self._vf.predict(
                self._vf_time(t), model.post_decision_state(state, x)
            )
            if value > best_value:
                best_x, best_value = x, value
        if best_x is None:
            raise ValueError(
                f"model.decision_space returned no feasible decision for {state!r}."
            )
        return best_x

    def fit(
        self,
        model: Any,
        horizon: int,
        n_iterations: int = 25,
        n_samples: int = 20,
        stepsize: float | None = None,
        random_state: RandomState = None,
    ) -> "ForwardADPPolicy":
        """Train the value function by simulating forward (figure 8.7).

        Runs ``n_iterations`` (index ``n``) of ``n_samples`` sample paths
        (index ``m``), each stepping ``t = 0..horizon-1``. After each path the
        realized path costs ``v̂_t`` are accumulated backward and smoothed into
        the value function.

        ``stepsize=None`` (default) uses a harmonic stepsize ``1/m`` where
        ``m`` counts *visits to each table entry* — the book's ``1/m`` rule
        applied per entry, so information accumulates across iterations
        instead of being overwritten at the start of each one. Pass a float
        for a constant stepsize.

        Returns ``self``; per-path total rewards are kept in
        ``training_rewards_`` (shape ``(n_iterations, n_samples)``) so learning
        can be monitored/plotted.
        """
        rng = check_random_state(
            random_state if random_state is not None else self.random_state
        )
        if self._vf is None:
            self._vf = TabularValueFunction(key=self.key)
        rewards = np.zeros((n_iterations, n_samples))

        for n in range(n_iterations):
            for m in range(1, n_samples + 1):
                state = model.initial_state()
                posts: list[Any] = []
                contribs: list[float] = []
                for t in range(horizon):
                    if self.exploration > 0.0 and rng.random() < self.exploration:
                        options = list(model.decision_space(state))
                        x = options[rng.integers(len(options))]
                    else:
                        x = self._argmax(state, model, t)
                    posts.append(model.post_decision_state(state, x))
                    exog = model.exogenous_info(state, x, rng)
                    contribs.append(float(model.objective(state, x, exog)))
                    state = model.transition(state, x, exog)

                # backward pass: v̂_t = C_t + v̂_{t+1}; update V at S^x_{t-1}
                v_hat = 0.0
                for t in reversed(range(horizon)):
                    v_hat = contribs[t] + v_hat
                    if t >= 1:
                        self._vf.update(self._vf_time(t - 1), posts[t - 1], v_hat, alpha=stepsize)
                rewards[n, m - 1] = sum(contribs)

        self.training_rewards_ = rewards
        self._fitted = True
        return self

    def decide(self, state: Any, model: Any = None, t: int | None = None) -> Any:
        if not self._fitted or self._vf is None:
            raise NotFittedError(
                "ForwardADPPolicy must be fitted before deciding: call "
                "policy.fit(model, horizon=...) first (or pass a trained value_function)."
            )
        if model is None:
            raise ValueError("ForwardADPPolicy needs the model to enumerate decisions.")
        return self._argmax(state, model, t)

    def tags(self) -> dict[str, Any]:
        return {"policy_class": "VFA", "needs_model": True, "trained": self._fitted}
