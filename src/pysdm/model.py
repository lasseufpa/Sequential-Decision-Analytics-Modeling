"""The ``Model``: the five elements of the Universal Modeling Framework.

A ``Model`` describes *the problem* — never the strategy used to solve it.
Subclasses implement four required methods (plus optional hooks)::

    class InventoryModel(sd.Model):
        def initial_state(self):                              # S_0
            ...
        def exogenous_info(self, state, decision, rng):       # W_{t+1}
            ...
        def transition(self, state, decision, exog_info):     # S_{t+1} = S^M(S_t, x_t, W_{t+1})
            ...
        def objective(self, state, decision, exog_info):      # C_t(S_t, x_t, W_{t+1})
            ...

Foolproofing: subclasses are checked **at class definition time** — a missing
method or a wrong number of parameters raises :class:`ModelDefinitionError`
with the expected signature spelled out, instead of a cryptic ``TypeError``
1000 steps into a simulation.

``exogenous_info`` is the model's *internal* description of uncertainty (a
mathematical sampling model). To drive a simulation with an external dataset
or a live feed instead, pass an :class:`~pysdm.exogenous.ExogenousSource` to
the :class:`~pysdm.engine.Engine` — the model itself never changes.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np

from .exceptions import ModelDefinitionError

__all__ = ["Model"]

# method name -> (parameters after `self`, human-readable role)
_REQUIRED_SIGNATURES: dict[str, tuple[tuple[str, ...], str]] = {
    "initial_state": ((), "S_0 — the initial state"),
    "exogenous_info": (("state", "decision", "rng"), "W_{t+1} — sampled exogenous information"),
    "transition": (("state", "decision", "exog_info"), "S_{t+1} — the transition function S^M"),
    "objective": (("state", "decision", "exog_info"), "C_t — the contribution function"),
}

_OPTIONAL_SIGNATURES: dict[str, tuple[tuple[str, ...], str]] = {
    "decision_space": (("state",), "feasible decisions in a state"),
    "post_decision_state": (("state", "decision"), "S^x_t — the post-decision state"),
    "contribution": (("state", "decision"), "C_t(S_t, x_t) — deterministic contribution"),
}


def _check_signature(cls_name: str, name: str, func: Any, expected: tuple[str, ...]) -> None:
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):  # builtins/C functions — nothing to check
        return
    # accept *args/**kwargs implementations
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        return
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "self"
    ]
    required = [p for p in positional if p.default is inspect.Parameter.empty]
    if len(required) > len(expected) or len(positional) < len(expected):
        wanted = ", ".join(("self",) + expected)
        raise ModelDefinitionError(
            f"{cls_name}.{name} has signature ({', '.join(p.name for p in params)}) "
            f"but pysdm will call it as {name}({wanted}). "
            f"Adjust the method to accept exactly these parameters."
        )


class Model(ABC):
    """Contract for a sequential decision problem (the "rules of the game").

    Required: :meth:`initial_state`, :meth:`exogenous_info`, :meth:`transition`,
    :meth:`objective`. Optional hooks used by optimizing/learning policies:
    :meth:`decision_space`, :meth:`post_decision_state`, :meth:`contribution`.

    Conventions:

    * ``transition`` must **return a new state**, never mutate the one it
      received — lookahead policies and parallel runs rely on this.
    * ``objective`` returns a plain ``float`` (reward convention: higher is
      better; encode costs as negative contributions).
    * ``__init__`` should only store parameters (no I/O, no heavy validation).
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        checks = dict(_REQUIRED_SIGNATURES)
        checks.update(_OPTIONAL_SIGNATURES)
        for name, (expected, _role) in checks.items():
            func = cls.__dict__.get(name)
            if func is None or getattr(func, "__isabstractmethod__", False):
                continue
            if isinstance(func, (staticmethod, classmethod)):
                func = func.__func__
            _check_signature(cls.__name__, name, func, expected)

    # ------------------------------------------------------------------ #
    # The five elements (state itself is declared via sd.State)          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def initial_state(self) -> Any:
        """Return the initial state ``S_0``."""

    @abstractmethod
    def exogenous_info(self, state: Any, decision: Any, rng: np.random.Generator) -> Any:
        """Sample ``W_{t+1}`` — the information revealed after deciding.

        May depend on ``state`` and ``decision`` (e.g. market impact). This is
        the model's *assumed* uncertainty; the engine can replace it with real
        data via an ``ExogenousSource`` without touching this method.
        """

    @abstractmethod
    def transition(self, state: Any, decision: Any, exog_info: Any) -> Any:
        """Return the next state ``S_{t+1} = S^M(S_t, x_t, W_{t+1})`` (a new object)."""

    @abstractmethod
    def objective(self, state: Any, decision: Any, exog_info: Any) -> float:
        """Return the contribution ``C_t(S_t, x_t, W_{t+1})`` of this step."""

    # ------------------------------------------------------------------ #
    # Optional hooks for optimizing / learning policies                  #
    # ------------------------------------------------------------------ #

    def decision_space(self, state: Any) -> Iterable[Any]:
        """Enumerate the feasible decisions in ``state``.

        Required by policies that take an argmax over decisions (e.g.
        :class:`~pysdm.policies.ForwardADPPolicy`). Default: not implemented.
        """
        raise ModelDefinitionError(
            f"{type(self).__name__} does not implement decision_space(state). "
            f"Policies that enumerate decisions (lookahead/ADP) require it: "
            f"return an iterable of feasible decisions for the given state."
        )

    def post_decision_state(self, state: Any, decision: Any) -> Any:
        """Return the post-decision state ``S^x_t`` — the state right after
        deciding but *before* new information arrives.

        Required by value-function policies (section 8.6.4 of the book).
        Example (energy storage): ``S_t=(R_t, p_t)`` gives ``S^x_t=(R_t+x, p_t)``.
        """
        raise ModelDefinitionError(
            f"{type(self).__name__} does not implement post_decision_state(state, decision). "
            f"Value-function policies (e.g. ForwardADPPolicy) require it."
        )

    def contribution(self, state: Any, decision: Any) -> float:
        """Deterministic part of the objective, ``C_t(S_t, x_t)``, evaluable
        *before* ``W_{t+1}`` is known. Used by lookahead/ADP policies.

        Default: delegates to ``objective(state, decision, None)``; override if
        your objective cannot handle ``exog_info=None``.
        """
        return self.objective(state, decision, None)
