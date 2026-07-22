"""Step-by-step record of everything that happened during a run.

Every step produces a :class:`StepRecord` — episode, time, state, decision,
exogenous information, contribution and timing. The :class:`History` collects
them and exports to a tidy ``pandas.DataFrame`` (declarative elements are
expanded into one column per field: ``state.resource``, ``exog.change``...),
so plots and custom analyses never require touching engine internals.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterator

__all__ = ["StepRecord", "History"]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One step of one episode: ``(S_t, x_t, W_{t+1}, C_t, S_{t+1})``."""

    episode: int
    t: int
    state: Any
    decision: Any
    exog_info: Any
    contribution: float
    next_state: Any
    decision_time: float  # seconds spent inside policy.decide


def _expand(prefix: str, obj: Any, row: dict[str, Any]) -> None:
    """Flatten declarative elements/dataclasses into ``prefix.field`` columns."""
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        items = to_dict()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        items = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    else:
        row[prefix] = obj
        return
    for name, value in items.items():
        row[f"{prefix}.{name}"] = value


class History:
    """Append-only container of :class:`StepRecord`.

    Recording can be disabled on the engine (``record_history=False``) for
    very large runs where only aggregated metrics matter.
    """

    def __init__(self) -> None:
        self.records: list[StepRecord] = []

    def append(self, record: StepRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[StepRecord]:
        return iter(self.records)

    def episode(self, k: int) -> list[StepRecord]:
        """All records of episode ``k``, in time order."""
        return [r for r in self.records if r.episode == k]

    def to_frame(self, include_next_state: bool = False):
        """Export as a long-format ``pandas.DataFrame`` (one row per step)."""
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError(
                "History.to_frame requires pandas: pip install 'pysdm[analysis]'."
            ) from err
        rows = []
        for r in self.records:
            row: dict[str, Any] = {"episode": r.episode, "t": r.t}
            _expand("state", r.state, row)
            _expand("decision", r.decision, row)
            _expand("exog", r.exog_info, row)
            row["contribution"] = r.contribution
            row["decision_time_ms"] = r.decision_time * 1e3
            if include_next_state:
                _expand("next_state", r.next_state, row)
            rows.append(row)
        return pd.DataFrame(rows)
