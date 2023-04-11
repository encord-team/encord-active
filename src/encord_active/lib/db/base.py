from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataUnit:
    hash: str
    location: Path
    title: str
    frame: int

    def __post_init__(self):
        # enforce conversion from probable str to Path
        self.location = Path(self.location)
