from __future__ import absolute_import, annotations

from functools import wraps
from typing import (
    Callable,
    Dict,
    NamedTuple,
    Protocol,
    Tuple,
    TypeAlias,
    runtime_checkable,
)

import numpy as np
from elfi import Discrepancy, ElfiModel, GPyRegression
from numpy.random import RandomState
from numpy.typing import NDArray

Bounds: TypeAlias = Dict[str, Tuple[float, ...]]


class ELFIModelBuilder(Protocol):
    def build_model(self) -> ModelAndDiscrepancy: ...


class ELFIInferenceProblem(ELFIModelBuilder):
    @property
    def name(self): ...

    @property
    def bounds(self) -> Bounds: ...

    @property
    def constraint(self): ...


@runtime_checkable
class SupportsBuildTargetModel(Protocol):
    def build_target_model(self, model: ElfiModel) -> GPyRegression: ...


class ModelAndDiscrepancy(NamedTuple):
    model: ElfiModel
    discrepancy: Discrepancy


class SimulatorFunction[*T](Protocol):
    def __call__(
        self,
        *args: *T,
        batch_size: int,
        random_state: RandomState,
    ) -> NDArray: ...


def with_constraint[
    *T
](
    inner: SimulatorFunction[*T], constraint: Callable[[*T], NDArray[np.bool_]]
) -> SimulatorFunction[*T]:
    @wraps(inner)
    def wrapper(*args: *T, **kwargs):
        out = inner(*args, **kwargs)
        failed = ~constraint(*args)
        out[failed] = np.nan
        return out

    return wrapper
