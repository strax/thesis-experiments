from copy import deepcopy
from numpy.random import SeedSequence

def seed2int(seed: SeedSequence):
    return seed.generate_state(1).item()

def split_seed(seed: SeedSequence, n: int):
    assert n > 0
    return map(deepcopy, seed.spawn(n))

__all__ = ["seed2int", "split_seed"]
