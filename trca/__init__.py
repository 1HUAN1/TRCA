"""Paper-aligned Transition-wise Rubric Credit Assignment (TRCA)."""

from trca.rewards import compute_trca_rewards
from trca.schema import RubricLibrary, RubricOperator

__all__ = ["RubricLibrary", "RubricOperator", "compute_trca_rewards"]
