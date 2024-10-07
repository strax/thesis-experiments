import time
from dataclasses import dataclass, field
from datetime import timedelta


@dataclass(frozen=True)
class Timer:
    begin: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> timedelta:
        return timedelta(seconds=time.monotonic() - self.begin)

