from pathlib import Path
from typing import NamedTuple


class DataUnit(NamedTuple):
    hash: str
    group_hash: str
    location: str
    title: str
    frame: int

    @property
    def location_(self) -> Path:
        return Path(self.location)
