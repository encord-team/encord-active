from abc import ABC
from typing import Optional, Set, Tuple, Dict

from torch import ByteTensor


class AnalysisStep(ABC):
    def __init__(self,
                 ident: str,
                 dependencies: Set[str],
                 temporal: Optional[Tuple[int, int]]) -> None:
        self.ident = ident
        self.dependencies = dependencies
        self.temporal = temporal

    def execute(self,
                image: ByteTensor,
                objects: Dict[str, object],
                dependencies: object) -> None:
        ...


class DerivedMetricStep(ABC):
    def __init__(self,
                 ident: str,
                 dependencies: Set[str],
                 temporal: Optional[Tuple[int, int]]) -> None:
        self.ident = ident
        self.dependencies = dependencies
        self.temporal = temporal

    def execute(self, dependencies: object) -> object:
        ...


class BaseEvaluation(ABC):
    def __init__(self, ident: str, dependencies: Set[str]) -> None:
        self.ident = ident
        self.dependencies = dependencies


class BaseAnalysis(BaseEvaluation):
    def __init__(self, ident: str, dependencies: Set[str], long_name: str, short_desc: str, long_desc: str) -> None:
        super().__init__(self, ident, dependencies)
        self.long_name = long_name
        self.short_desc = short_desc
        self.long_desc = long_desc


class TemporalBaseAnalysis(BaseAnalysis):
    def __init__(self, ident: str, dependencies: Set[str], long_name: str, short_desc: str, long_desc: str,
                 prev_frame_count: int, next_frame_count: int):
        super().__init__(ident, dependencies, long_name, short_desc, long_desc)
        self.prev_frame_count = prev_frame_count
        self.next_frame_count = next_frame_count
