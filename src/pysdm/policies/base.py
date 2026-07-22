"""The single ``Policy`` contract.

There is exactly **one** base class. The book's four meta-classes (PFA, CFA,
VFA, DLA) are *categories*, not code: a concrete policy declares its category
in :meth:`Policy.tags`, and the engine never cares which one it is.

Foolproofing — attribute requirements
-------------------------------------
Policies read specific attributes off the state (e.g. a UCB policy needs
``state.mu`` and ``state.counts``). Each policy declares them in
``state_requirements``; the engine verifies them against ``S_0`` *before*
running and raises :class:`MissingStateAttributeError` naming exactly what is
missing — instead of an ``AttributeError`` from the middle of a rollout.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..exceptions import MissingStateAttributeError

__all__ = ["Policy"]


class Policy(ABC):
    """Decision rule ``X^π(S_t) -> x_t``.

    Required: :meth:`decide`. Optional: :meth:`update` (for learning policies),
    :meth:`reset` (per-episode internal state), :meth:`tags` (qualitative
    metadata). ``get_params``/``set_params`` follow the sklearn convention and
    are derived automatically from ``__init__``.
    """

    #: State attributes this policy reads. Verified by the engine before a run.
    state_requirements: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def decide(self, state: Any, model: Any = None) -> Any:
        """Return the decision ``x_t`` for ``state``.

        ``model`` is provided for policies that look inside the problem
        (lookahead, ADP); simple rule-based policies ignore it.
        """

    def update(self, t: int, state: Any, decision: Any, exog_info: Any, next_state: Any) -> None:
        """Hook called by the engine after every transition.

        Override in policies that learn online (value functions, beliefs kept
        inside the policy). Default: no-op.
        """

    def reset(self) -> None:
        """Hook called by the engine at the start of each episode. Default: no-op."""

    # ------------------------------------------------------------------ #
    # Foolproofing                                                        #
    # ------------------------------------------------------------------ #

    def validate_state(self, state: Any) -> None:
        """Raise if ``state`` lacks attributes listed in ``state_requirements``."""
        missing = [a for a in self.state_requirements if not hasattr(state, a)]
        if missing:
            raise MissingStateAttributeError(
                f"{type(self).__name__} requires state attribute(s) "
                f"{', '.join(repr(m) for m in missing)}, not found on "
                f"{type(state).__name__}. Add the field(s) to your State class "
                f"or choose a policy compatible with it."
            )

    # ------------------------------------------------------------------ #
    # sklearn-style parameter handling (enables grid search / cloning)    #
    # ------------------------------------------------------------------ #

    @classmethod
    def _param_names(cls) -> list[str]:
        init = cls.__init__
        if init is object.__init__:
            return []
        return [
            name
            for name, p in inspect.signature(init).parameters.items()
            if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]

    def get_params(self) -> dict[str, Any]:
        """Constructor parameters as a dict (sklearn convention)."""
        return {name: getattr(self, name) for name in self._param_names() if hasattr(self, name)}

    def set_params(self, **params: Any) -> "Policy":
        """Set constructor parameters in place; returns ``self``."""
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Unknown parameter {name!r} for {type(self).__name__}. "
                    f"Valid parameters: {sorted(valid)}."
                )
            setattr(self, name, value)
        return self

    def tags(self) -> dict[str, Any]:
        """Qualitative metadata (policy class PFA/CFA/VFA/DLA, complexity...)."""
        return {}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"
