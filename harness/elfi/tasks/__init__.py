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

def with_constraint[*T](
    inner: SimulatorFunction[*T], constraint: Callable[[*T], NDArray[np.bool_]]
) -> SimulatorFunction[*T]:
    @wraps(inner)
    def wrapper(*args: *T, batch_size: int, random_state: RandomState):
        simulated = inner(*args, batch_size=batch_size, random_state=random_state)
        theta = np.stack(args, axis=-1)
        p_feasible = constraint(theta)
        # Sample Bernoulli rvs for each p_feasible
        # NOTE: Because `random_sample` returns values in the half-open range 0 <= x < 1,
        #       for deterministic constraints this step is simply a no-op.
        feasible = random_state.random_sample(np.shape(p_feasible)) < p_feasible
        return np.where(
            feasible,
            simulated,
            np.nan
        )

    return wrapper
