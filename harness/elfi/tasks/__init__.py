from __future__ import absolute_import, annotations

from typing import (
    Dict,
    Callable,
    Tuple,
    TypeAlias,
    TypedDict,
    Protocol,
    NotRequired,
    runtime_checkable,
)
from dataclasses import dataclass
from functools import wraps
from numpy.random import RandomState
from numpy.typing import NDArray

from elfi import ElfiModel, NodeReference, GPyRegression

import numpy as np

Bounds: TypeAlias = Dict[str, Tuple[float, ...]]


class ELFIModelBuilder(Protocol):
    @property
    def bounds(self) -> Bounds: ...

    @property
    def constraint(self): ...

    def build_model(self) -> ModelBundle: ...


class ELFIInferenceProblem(ELFIModelBuilder):
    @property
    def name(self):
        pass

@runtime_checkable
class SupportsBuildTargetModel(Protocol):
    def build_target_model(self, model: ElfiModel) -> GPyRegression:
        ...


@dataclass
class ModelBundle:
    model: ElfiModel
    target: NodeReference


class SimulatorContext(TypedDict):
    batch_size: NotRequired[int | None]
    random_state: NotRequired[RandomState | None]


class SimulatorFunction[*T](Protocol):
    def __call__(
        self,
        *args: *T,
        batch_size: int | None = ...,
        random_state: RandomState | None = ...,
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


def with_stochastic_failures[
    *T
](inner: SimulatorFunction[*T], *, p: float) -> SimulatorFunction[*T]:
    @wraps(inner)
    def wrapped(*params: *T, batch_size=1, random_state: RandomState):
        assert isinstance(random_state, RandomState)

        out = inner(*params, batch_size=batch_size, random_state=random_state)
        failed = random_state.random(batch_size) <= p
        out[failed] = np.nan
        return out

    return wrapped
