from enum import StrEnum, auto

class FeasibilityEstimatorKind(StrEnum):
    NONE = auto()
    ORACLE = auto()
    GPC_MATERN = auto()

__all__ = []
