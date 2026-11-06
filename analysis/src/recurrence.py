"""Contact recurrence over sampled trajectory frames.

Contacts are unordered particle pairs or qualifying molecule pairs. The score
is 1 - n_distinct_contacts / n_contact_observations in the selected window.
"""
from collections import deque
from collections.abc import Iterable


def pair_set(left: Iterable[int], right: Iterable[int]) -> set[tuple[int, int]]:
    """Normalize an array of contact endpoints into unique unordered pairs."""
    return {(min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(left, right) if int(a) != int(b)}


def recurrence_from_contact_sets(contact_sets: Iterable[set[tuple[int, int]]]) -> float:
    """Return contact recurrence, or NaN if the window contains no contacts."""
    sets = list(contact_sets)
    n_observations = sum(len(s) for s in sets)
    if not n_observations:
        return float("nan")
    n_unique = len(set().union(*sets))
    return 1.0 - n_unique / n_observations


class RollingRecurrence:
    """Stateful rolling recurrence; the first (window-1) frames are warmup."""

    def __init__(self, window: int = 20) -> None:
        if window < 2:
            raise ValueError("recurrence window must be >= 2 frames")
        self.window = window
        self._sets: deque[set[tuple[int, int]]] = deque(maxlen=window)

    def add(self, contact_set: set[tuple[int, int]]) -> float:
        self._sets.append(contact_set)
        if len(self._sets) < self.window:
            return float("nan")
        return recurrence_from_contact_sets(self._sets)
