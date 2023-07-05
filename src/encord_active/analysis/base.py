from abc import ABC, abstractmethod
from typing import Any

from encord_active.analysis.types import (
    AnnotationsMetricResult,
    ClassificationMetadata,
    ImageTensor,
    MaskTensor,
    MetricDependencies,
    MetricKey,
    MetricResult,
    ObjectMetadata,
)

# class AnalysisStep(ABC):
#     def __init__(self, ident: str, dependencies: set[str], temporal: tuple[int, int] | None) -> None:
#         self.ident = ident
#         self.dependencies = dependencies
#         self.temporal = temporal
#
#     def execute(self, image: ImageTensor, objects: dict[str, object], dependencies: object) -> None:
#         # TODO type dependencies
#         # TODO objects should probably be `ObjectMetadata` from the types file
#         ...
#

# class DerivedMetricStep(ABC):
#     def __init__(self, ident: str, dependencies: set[str], temporal: tuple[int, int] | None) -> None:
#         self.ident = ident
#         self.dependencies = dependencies
#         self.temporal = temporal
#
#     def execute(self, dependencies: object) -> object:
#         # TODO type dependencies
#         ...
#
#


class BaseEvaluation(ABC):
    """
    Base evaluations are the most abstract definitions of computations that need to happen.
    They can be both:

    - Quality Metrics that needs to be stored (The `BaseAnalysis` subclass)
    - Computations the need to happen for Quality Metrics to be able to share computations.
      For example, Embeddings, image transforms, etc.

    """

    def __init__(self, ident: str, dependencies: set[str]) -> None:
        self.ident = ident
        self.dependencies = dependencies

    @abstractmethod
    def _calculate(
        self,
        image: ImageTensor,
        image_deps: MetricDependencies,
        obj: ObjectMetadata | None,
        obj_deps: MetricDependencies | None,
        clf: ClassificationMetadata | None,
        clf_deps: MetricDependencies | None,
        objects: dict[MetricKey, ObjectMetadata],
        classifications: dict[MetricKey, ClassificationMetadata],
        **kwargs,
    ) -> MetricResult | AnnotationsMetricResult:
        # All arguments should be available here, such that we can call the same thing for all metrics.
        # Every sub-class will route to a function taking only the relevant arguments for that particular type of analysis
        # TODO neighbouring frames not
        ...


class BaseAnalysis(BaseEvaluation):
    """
    The `BaseAnalysis` is all the metrics that needs to be stored in the
    database.
    """

    def __init__(self, ident: str, dependencies: set[str], long_name: str, desc: str) -> None:
        super().__init__(ident, dependencies)
        self.long_name = long_name
        self.desc = desc


class TemporalBaseAnalysis(BaseAnalysis):
    """
    The `TemporalBaseAnalysis` is all the metrics that require information from
    neighbouring frames in, e.g., image groups of videos and needs to be stored
    in the database.
    """

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

    @abstractmethod
    def _calculate(
        self,
        image: ImageTensor,
        image_deps: MetricDependencies,
        obj: ObjectMetadata | None,
        obj_deps: MetricDependencies | None,
        clf: ClassificationMetadata | None,
        clf_deps: MetricDependencies | None,
        objects: dict[MetricKey, ObjectMetadata],
        classifications: dict[MetricKey, ClassificationMetadata],
        prev_frames: list[Any],
        next_frames: list[Any],
    ) -> MetricResult | AnnotationsMetricResult:
        # All arguments should be available here, such that we can call the same thing for all metrics.
        # Every sub-class will route to a function taking only the relevant arguments for that particular type of analysis
        # TODO neighbouring frames not
        ...
