from __future__ import absolute_import, annotations

from abc import ABC
from typing import cast, Dict, Callable, Concatenate, Tuple, TypeAlias, TypedDict, Protocol, Unpack, NotRequired
from dataclasses import dataclass
from functools import wraps
from numpy.random import RandomState
from numpy.typing import NDArray

from elfi import ElfiModel, NodeReference

import numpy as np

Bounds: TypeAlias = Dict[str, Tuple[float, ...]]

class ELFIInferenceProblem(ABC):
    pass

class ELFIModelBuilder(Protocol):
    @property
    def bounds(self) -> Bounds:
        ...

    def build_model(self) -> ModelBundle:
        ...

@dataclass
class ModelBundle:
    model: ElfiModel
    target: NodeReference

class SimulatorContext(TypedDict):
    batch_size: NotRequired[int | None]
    random_state: NotRequired[RandomState | None]

class SimulatorFunction[*T](Protocol):
    def __call__(self, *args: *T, batch_size: int | None = ..., random_state: RandomState | None = ...) -> NDArray:
        ...

def with_constraint[*T](inner: SimulatorFunction[*T], constraint: Callable[[*T], NDArray[np.bool_]]) -> SimulatorFunction[*T]:
    @wraps(inner)
    def wrapper(*args: *T, **kwargs):
        out = inner(*args, **kwargs)
        failed = ~constraint(*args)
        out[failed] = np.nan
        return out

    return wrapper

def with_stochastic_failures[*T](inner: SimulatorFunction[*T], *, p: float) -> SimulatorFunction[*T]:
    @wraps(inner)
    def wrapped(*params: *T, batch_size=1, random_state: RandomState):
        assert isinstance(random_state, RandomState)

        out = inner(*params, batch_size=batch_size, random_state=random_state)
        failed = random_state.random(batch_size) <= p
        out[failed] = np.nan
        return out

    return wrapped
