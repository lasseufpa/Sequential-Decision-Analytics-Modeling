"""Declarative building blocks of the Universal Modeling Framework.

The three data-carrying elements of a sequential decision problem — the state
``S_t``, the decision ``x_t`` and the exogenous information ``W_{t+1}`` — are
declared by *annotating fields* on a subclass. The base class turns the
subclass into a dataclass automatically, so this is all it takes::

    import pysdm as sd

    class StorageState(sd.State):
        resource: float
        price: float

    class PriceChange(sd.ExogenousInfo):
        change: float

    s = StorageState(resource=0.0, price=25.0)
    s2 = s.replace(price=26.5)      # states are treated as immutable snapshots
    s.to_dict()                     # {'resource': 0.0, 'price': 25.0}

Design notes:

* ``ExogenousInfo`` exists because ``W_{t+1}`` is, in general, a *vector* of
  simultaneous information sources (demand arrivals, price moves, equipment
  failures...), not a single number. Each source is one declared field.
* Scalars are still welcome: a model may return a plain ``float`` as its
  exogenous information. The declarative classes are for anything richer.
* Declaring a subclass with **no fields** is an immediate, loud error — a
  typo'd class body should fail at import time, not mid-simulation.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .exceptions import ModelDefinitionError

__all__ = ["State", "Decision", "ExogenousInfo"]


def _all_annotations(cls: type) -> dict[str, Any]:
    """Collect field annotations across the MRO (base-first)."""
    annotations: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        annotations.update(getattr(klass, "__annotations__", {}))
    return annotations


class _DeclarativeElement:
    """Shared machinery: auto-dataclass + convenience methods."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_pysdm_marker", False):
            return  # State/Decision/ExogenousInfo themselves, not user classes
        if not _all_annotations(cls):
            raise ModelDefinitionError(
                f"{cls.__name__} declares no fields. Declare each component as an "
                f"annotated attribute, e.g.:\n\n"
                f"    class {cls.__name__}({cls.__mro__[1].__name__}):\n"
                f"        resource: float\n"
            )
        if not dataclasses.is_dataclass(cls):
            # eq=False: fields may hold numpy arrays, whose == is elementwise
            # and would make the generated __eq__ raise.
            dataclasses.dataclass(eq=False)(cls)

    def replace(self, **changes: Any):
        """Return a copy of this element with ``changes`` applied.

        This is the idiomatic way to build ``S_{t+1}`` from ``S_t`` inside
        ``Model.transition`` without mutating the original.
        """
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Field name -> value mapping (shallow — array fields are not copied)."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Names of the declared components, in declaration order."""
        return tuple(f.name for f in dataclasses.fields(cls))


class State(_DeclarativeElement):
    """The state ``S_t``: everything known at time ``t`` needed to decide.

    May hold physical resources, information *and* beliefs (e.g. the mean and
    precision of a Bayesian estimate) — any annotated field works, including
    numpy arrays.
    """

    _pysdm_marker = True


class Decision(_DeclarativeElement):
    """The decision ``x_t``. Optional: plain ints/floats are fine for simple
    problems; declare a subclass when the decision has named components."""

    _pysdm_marker = True


class ExogenousInfo(_DeclarativeElement):
    """The exogenous information ``W_{t+1}``: what the world reveals *after*
    the decision. Declare one field per information source."""

    _pysdm_marker = True
