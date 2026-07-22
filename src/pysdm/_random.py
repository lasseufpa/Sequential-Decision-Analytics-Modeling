"""Reproducibility utilities."""

from __future__ import annotations

import numpy as np

RandomState = int | np.random.Generator | np.random.SeedSequence | None


def check_random_state(random_state: RandomState) -> np.random.Generator:
    """Normalize ``random_state`` into a :class:`numpy.random.Generator`.

    Accepts ``None`` (fresh entropy), an ``int`` seed, a ``SeedSequence`` or an
    existing ``Generator`` (returned as-is). This is the single entry point for
    randomness in the library, mirroring ``sklearn.utils.check_random_state``
    but built on the modern ``numpy.random.Generator`` API.
    """
    if random_state is None or isinstance(random_state, int):
        return np.random.default_rng(random_state)
    if isinstance(random_state, np.random.SeedSequence):
        return np.random.default_rng(random_state)
    if isinstance(random_state, np.random.Generator):
        return random_state
    raise TypeError(
        f"random_state must be None, int, SeedSequence or numpy Generator, "
        f"got {type(random_state).__name__!r}."
    )


def spawn_generators(random_state: RandomState, n: int) -> list[np.random.Generator]:
    """Derive ``n`` statistically independent generators from one seed.

    Used by the engine to give each episode (sample path) its own stream while
    keeping the whole run reproducible from a single ``random_state`` — this is
    what makes *common random numbers* comparisons possible: two runs with the
    same seed see exactly the same sample paths.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state.spawn(n)
    seq = random_state if isinstance(random_state, np.random.SeedSequence) else np.random.SeedSequence(random_state)
    return [np.random.default_rng(child) for child in seq.spawn(n)]
