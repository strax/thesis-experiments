from enum import StrEnum, auto

class FeasibilityEstimatorKind(StrEnum):
    NONE = auto()
    ORACLE = auto()
    GPC_MATERN52 = auto()

__all__ = []
