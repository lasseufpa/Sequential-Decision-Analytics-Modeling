"""Exception hierarchy for pysdm.

Every error raised by the library derives from :class:`PysdmError`, so users can
catch library problems with a single ``except PysdmError``. Errors carry
actionable messages: they say *what* is missing and *how* to fix it.
"""

from __future__ import annotations


class PysdmError(Exception):
    """Base class for all pysdm errors."""


class ModelDefinitionError(PysdmError, TypeError):
    """A ``Model`` (or declarative element) subclass is malformed.

    Raised at *class definition time* whenever possible, so mistakes surface
    immediately instead of deep inside a simulation run.
    """


class MissingStateAttributeError(PysdmError, AttributeError):
    """The state object does not expose an attribute a policy requires.

    Policies declare their needs in ``Policy.state_requirements``; the engine
    verifies them against the initial state before running.
    """


class InvalidDecisionError(PysdmError, ValueError):
    """A policy produced a decision the model considers invalid."""


class ExogenousSourceError(PysdmError, RuntimeError):
    """An exogenous data source failed or was exhausted."""


class NotFittedError(PysdmError, RuntimeError):
    """A learning policy was used before calling ``fit``."""
