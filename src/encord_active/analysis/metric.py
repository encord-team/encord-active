from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Set, Dict

from encord_active.analysis.base import BaseAnalysis, BaseFrameInput, BaseFrameOutput
from encord_active.analysis.types import (
    AnnotationMetadata,
    ImageTensor,
    MaskTensor,
    MetricDependencies,
    MetricKey,
    MetricResult,
)
from encord_active.db.models import AnnotationType

class BaseAnalysisWithAnnotationFilter(BaseAnalysis, metaclass=ABCMeta):
    """
    Simple metric that only depends on the pixels, if enabled the mask argument allows this metric to
    also apply to objects by only considering the subset of the image the object is present in.
    """

    def __init__(
        self,
        ident: str,
        dependencies: set[str],
        long_name: str,
        desc: str,
        annotation_types: Optional[Set[AnnotationType]] = None,
    ) -> None:
        super().__init__(ident, dependencies, long_name, desc)
        self.annotation_types = annotation_types


class OneImageMetric(BaseAnalysisWithAnnotationFilter, metaclass=ABCMeta):
    """
    Simple metric that only depends on the pixels, if enabled the mask argument allows this metric to
    also apply to objects by only considering the subset of the image the object is present in.
    """

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        image = self.calculate(frame.image_deps, frame.image, None)
        annotations = {}
        for annotation_hash, annotation in frame.annotations.items():
            if self.annotation_types is not None and annotation.annotation_type not in self.annotation_types:
                continue
            annotation_deps = frame.annotations_deps[annotation_hash]
            if annotation.annotation_type != AnnotationType.CLASSIFICATION:
                annotations[annotation_hash] = self.calculate(
                    deps=annotation_deps,
                    image=frame.image,
                    mask=annotation.mask,
                )
            else:
                annotations[annotation_hash] = image
        return BaseFrameOutput(
            image=image,
            annotations=annotations
        )

    @abstractmethod
    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: MaskTensor | None) -> MetricResult:
        ...


class OneObjectMetric(BaseAnalysisWithAnnotationFilter, metaclass=ABCMeta):
    """
    Simple metric that only depends on the object shape itself (& pixels via per-object embeddings)
    This is implicitly applied to all child classifications of an object.
    """

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        annotations = {}
        for annotation_hash, annotation in frame.annotations.items():
            if self.annotation_types is not None and annotation.annotation_type not in self.annotation_types:
                continue
            annotation_deps = frame.annotations_deps[annotation_hash]
            annotations[annotation_hash] = self.calculate(
                annotation=annotation,
                deps=annotation_deps,
            )
        return BaseFrameOutput(
            image=None,
            annotations=annotations,
        )

    @abstractmethod
    def calculate(self, annotation: AnnotationMetadata, deps: MetricDependencies) -> MetricResult:
        ...


class ObjectByFrameMetric(BaseAnalysis, metaclass=ABCMeta):
    """
    More complex single frame object metric where the result of each object depends on all other objects
    present in the single frame. The keys of the result should match the keys of 'objs'.
    """

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        return BaseFrameOutput(
            image=None,
            annotations=self.calculate(
                frame.annotations,
                frame.annotations_deps,
            )
        )

    @abstractmethod
    def calculate(
        self,
        annotations: Dict[str, AnnotationMetadata],
        annotation_deps: dict[str, MetricDependencies],
    ) -> Dict[str, MetricResult]:
        ...


class ImageObjectsMetric(BaseAnalysis, metaclass=ABCMeta):

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        return BaseFrameOutput(
            image=self.calculate(),
            annotations={},
        )

    @abstractmethod
    def calculate(
        self, img_deps: MetricDependencies, obj_deps: dict[str, MetricDependencies], objs: dict[str, ObjectMetadata]
    ) -> MetricResult:
        """
        TODO: This is currently only used by object count which doesn't require all these arguments.
        """
        ...


class TemporalOneImageMetric(BaseAnalysisWithAnnotationFilter, metaclass=ABCMeta):
    """
    Temporal variant of [OneImageMetric]
    """

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        if prev_frame is None or next_frame is None:
            return None

    @abstractmethod
    def calculate_default(self) -> MetricResult:
        ...

    @abstractmethod
    def calculate(
        self,
        deps: MetricDependencies,
        image: ImageTensor,
        mask: Optional[MaskTensor],
        prev_image: ImageTensor,
        prev_mask: Optional[MaskTensor],
        next_image: ImageTensor,
        next_mask: Optional[MaskTensor],
    ) -> MetricResult:
        ...


class TemporalOneObjectMetric(BaseAnalysis, metaclass=ABCMeta):
    """
    Temporal variant of [OneObjectMetric].
    """

    @abstractmethod
    def calculate(
        self,
        deps: MetricDependencies,
        obj: ObjectMetadata,
        prev_frames: list[tuple[MetricDependencies, ObjectMetadata]],
        next_frames: list[tuple[MetricDependencies, ObjectMetadata]],
    ) -> MetricResult:
        ...


class TemporalObjectByFrameMetric(TemporalBaseAnalysis, metaclass=ABCMeta):
    @abstractmethod
    def calculate(
        self,
        img_deps: MetricDependencies,
        obj_deps: dict[str, MetricDependencies],
        objs: dict[str, ObjectMetadata],
        prev_frames: list[tuple[MetricDependencies, dict[str, MetricDependencies], dict[str, ObjectMetadata]]],
        next_frames: list[tuple[MetricDependencies, dict[str, MetricDependencies], dict[str, ObjectMetadata]]],
    ) -> dict[str, MetricResult]:
        ...


class DerivedMetric(BaseAnalysisWithAnnotationFilter, metaclass=ABCMeta):
    """
    Simple metric that only depends on the pixels, if enabled the mask argument allows this metric to
    also apply to objects by only considering the subset of the image the object is present in.
    """

    @abstractmethod
    def calculate(self, deps: MetricDependencies) -> MetricResult:
        ...
