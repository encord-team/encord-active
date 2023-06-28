from abc import ABC

from encord_active.analysis.types import ImageTensor


class AnalysisStep(ABC):
    def __init__(self, ident: str, dependencies: set[str], temporal: tuple[int, int] | None) -> None:
        self.ident = ident
        self.dependencies = dependencies
        self.temporal = temporal

    def execute(self, image: ImageTensor, objects: dict[str, object], dependencies: object) -> None:
        # TODO type dependencies
        # TODO objects should probably be `ObjectMetadata` from the types file
        ...


class DerivedMetricStep(ABC):
    def __init__(self, ident: str, dependencies: set[str], temporal: tuple[int, int] | None) -> None:
        self.ident = ident
        self.dependencies = dependencies
        self.temporal = temporal

    def execute(self, dependencies: object) -> object:
        # TODO type dependencies
        ...


class BaseEvaluation(ABC):
    def __init__(self, ident: str, dependencies: set[str]) -> None:
        self.ident = ident
        self.dependencies = dependencies


class BaseAnalysis(BaseEvaluation):
    def __init__(self, ident: str, dependencies: set[str], long_name: str, desc: str) -> None:
        super().__init__(ident, dependencies)
        self.long_name = long_name
        self.desc = desc


class TemporalBaseAnalysis(BaseAnalysis):
    def __init__(
        self,
        ident: str,
        dependencies: set[str],
        long_name: str,
        desc: str,
        prev_frame_count: int,
        next_frame_count: int,
    ):
        super().__init__(ident, dependencies, long_name, desc)
        self.prev_frame_count = prev_frame_count
        self.next_frame_count = next_frame_count
