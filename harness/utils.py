from typing import Callable

def maybe[T, U](func: Callable[[T], U], value: T | None, default: U) -> U:
    if value is None:
        return default
    return func(value)
