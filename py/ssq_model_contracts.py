"""Immutable configuration and contracts for per-ball models."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

MODEL_TRAINING_PARAMS = MappingProxyType({
    'random_state': 42,
    'deterministic': True,
    'force_col_wise': True,
    'verbose': -1,
})


@dataclass(frozen=True)
class BallModelSpec:
    candidates: Sequence[int]
    outcome_column: str
    contains_candidate: Callable[[object, int], bool]
    description: str | None = None
