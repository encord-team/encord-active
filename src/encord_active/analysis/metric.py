from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from encord_active.analysis.base import BaseAnalysis, TemporalBaseAnalysis
from encord_active.analysis.types import (
    AnnotationsMetricDependencies,
    AnnotationsMetricResult,
    ClassificationMetadata,
    ImageTensor,
    MaskTensor,
    MetricDependencies,
    MetricKey,
    MetricResult,
    ObjectMetadata,
)


@dataclass
class MetricConfig:
    """
    version: version of the metric
    ranked: if true then the exported float should be treated as a score
            and transformed implicitly to the index under ORDER BY ASC
            instead. (smallest value is rank 1, largest value is rank N)
    """

    version: int
    ranked: bool
    # FIXME: groupings (float => string enum)


class OneImageMetric(BaseAnalysis, metaclass=ABCMeta):
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
        apply_to_objects: bool = True,
        apply_to_classifications: bool = True,
    ) -> None:
        super().__init__(ident, dependencies, long_name, desc)
        self.apply_to_objects = apply_to_objects
        self.apply_to_classifications = apply_to_classifications

    def _calculate(
        self,
        image: ImageTensor,
        image_deps: MetricDependencies,
        obj: ObjectMetadata | None,
        obj_deps: MetricDependencies | None,
        clf: ClassificationMetadata | None,
        clf_deps: MetricDependencies | None,
        objects: dict[MetricKey, ObjectMetadata],
        objects_deps: AnnotationsMetricResult | None,
        classifications: dict[MetricKey, ClassificationMetadata],
        classifications_deps: AnnotationsMetricResult | None,
        **kwargs,
    ) -> MetricResult | AnnotationsMetricResult:
        mask = None
        deps = image_deps
        if obj is not None:
            assert obj_deps is not None
            mask = obj.mask
            deps = obj_deps
        return self.calculate(deps, image, mask)

    @abstractmethod
    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: MaskTensor | None) -> MetricResult:
        pass


class OneObjectMetric(BaseAnalysis, metaclass=ABCMeta):
    """
    Simple metric that only depends on the object shape itself (& pixels via per-object embeddings)
    This is implicitly applied to all child classifications of an object.
    """

    def _calculate(
        self,
        image: ImageTensor,
        image_deps: MetricDependencies,
        obj: ObjectMetadata | None,
        obj_deps: MetricDependencies | None,
        clf: ClassificationMetadata | None,
        clf_deps: MetricDependencies | None,
        objects: dict[MetricKey, ObjectMetadata],
        objects_deps: AnnotationsMetricResult | None,
        classifications: dict[MetricKey, ClassificationMetadata],
        classifications_deps: AnnotationsMetricResult | None,
        **kwargs,
    ) -> MetricResult | AnnotationsMetricResult:
        if obj is None:
            return None
        return self.calculate(obj)

    @abstractmethod
    def calculate(self, obj: ObjectMetadata) -> MetricResult:
        pass


class ObjectByFrameMetric(BaseAnalysis, metaclass=ABCMeta):
    """
    More complex single frame object metric where the result of each object depends on all other objects
    present in the single frame. The keys of the result should match the keys of 'objs'.
    """

    def _calculate(
        self,
        image: ImageTensor,
        image_deps: MetricDependencies,
        obj: ObjectMetadata | None,
        obj_deps: MetricDependencies | None,
        clf: ClassificationMetadata | None,
        clf_deps: MetricDependencies | None,
        objects: dict[MetricKey, ObjectMetadata],
        objects_deps: AnnotationsMetricDependencies | None,
        classifications: dict[MetricKey, ClassificationMetadata],
        classifications_deps: AnnotationsMetricDependencies | None,
        **kwargs,
    ) -> MetricResult | AnnotationsMetricResult:
        if obj is not None or objects is None or objects_deps is None:
            return None

        return self.calculate(objects_deps, objects)

    @abstractmethod
    def calculate(
        self,
        obj_deps: AnnotationsMetricDependencies,
        objects: dict[MetricKey, ObjectMetadata],
    ) -> AnnotationsMetricResult:
        ...


class ImageObjectsMetric(BaseAnalysis, metaclass=ABCMeta):
    @abstractmethod
    def calculate(
        self, img_deps: MetricDependencies, obj_deps: dict[str, MetricDependencies], objs: dict[str, ObjectMetadata]
    ) -> MetricResult | AnnotationsMetricResult:
        """
        TODO: This is currently only used by object count which doesn't require all these arguments.
        """
        ...


class TemporalOneImageMetric(TemporalBaseAnalysis, metaclass=ABCMeta):
    """
    Temporal variant of [OneImageMetric]
    """

    def __init__(
        self,
        ident: str,
        dependencies: set[str],
        long_name: str,
        desc: str,
        prev_frame_count: int,
        next_frame_count: int,
        apply_to_objects: bool = True,
        apply_to_classifications: bool = True,
    ) -> None:
        super().__init__(ident, dependencies, long_name, desc, prev_frame_count, next_frame_count)
        self.apply_to_objects = apply_to_objects
        self.apply_to_classifications = apply_to_classifications

    # TODO Fix me

    @abstractmethod
    def calculate(
        self,
        deps: MetricDependencies,
        image: ImageTensor,
        mask: MaskTensor | None,
        prev_frames: list[tuple[MetricDependencies, ImageTensor, MaskTensor | None]],
        next_frames: list[tuple[MetricDependencies, ImageTensor, MaskTensor | None]],
    ) -> MetricResult:
        ...


class TemporalOneObjectMetric(TemporalBaseAnalysis, metaclass=ABCMeta):
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


class DerivedMetric(BaseAnalysis, metaclass=ABCMeta):
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
        apply_to_images: bool = True,
        apply_to_objects: bool = True,
        apply_to_classifications: bool = True,
    ) -> None:
        super().__init__(ident, dependencies, long_name, desc)
        self.apply_to_images = apply_to_images
        self.apply_to_objects = apply_to_objects
        self.apply_to_classifications = apply_to_classifications

    @abstractmethod
    def calculate(self, deps: MetricDependencies) -> MetricResult:
        ...
