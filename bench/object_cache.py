import pickle

from pathlib import Path
from dataclasses import dataclass
from hashlib import sha256

from .typing import StrPath

def _key_hash(key: str) -> str:
    return sha256(key.encode()).hexdigest()

@dataclass
class ObjectCache:
    """Simple persistent pickle-based object cache. Not thread-safe."""

    def __init__(self, path: StrPath):
        self.root = Path(path)

    def _object_path(self, key: str) -> Path:
        return self.root / _key_hash(key)

    def get(self, key: str) -> None | object:
        path = self._object_path(key)

        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def put(self, key: str, value: object):
        path = self._object_path(key)

        path.parent.mkdir(exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(value, f)
