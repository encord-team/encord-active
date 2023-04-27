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
    data_hash: Optional[str] = None
    data_title: Optional[str] = None
    frame: Optional[int] = None
    location: Optional[str] = None
    lr_data_hash: Optional[str] = None
