from pathlib import Path
from typing import NamedTuple, Optional


class DataUnit(NamedTuple):
    hash: str
    group_hash: str
    location: str
    title: str
    frame: int

    @property
    def location_(self) -> Path:
        return Path(self.location)


class DataUnitLike(NamedTuple):
    hash: Optional[str] = None
    group_hash: Optional[str] = None
    location: Optional[str] = None
    title: Optional[str] = None
    frame: Optional[int] = None
