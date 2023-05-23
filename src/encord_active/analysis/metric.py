from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Set, Optional, Dict, Union
from dataclasses import dataclass
from torch import ByteTensor, BoolTensor, FloatTensor

from encord_active.analysis.base import BaseAnalysis, TemporalBaseAnalysis
from encord_active.analysis.types import ObjectMetadata


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


def image_width(image: ByteTensor) -> int:
    return image.shape[1]


def image_height(image: ByteTensor) -> int:
    return image.shape[0]


MetricDependencies = Dict[str, Union[FloatTensor, float]]


class OneImageMetric(BaseAnalysis, metaclass=ABCMeta):
    """
    Simple metric that only depends on the pixels, if enabled the mask argument allows this metric to
    also apply to objects by only considering the subset of the image the object is present in.
    """

    def __init__(self, ident: str, dependencies: Set[str], long_name: str, short_desc: str, long_desc: str,
                 apply_to_objects: bool = True, apply_to_classifications: bool = True) -> None:
        super().__init__(ident, dependencies, long_name, short_desc, long_desc)
        self.apply_to_objects = apply_to_objects
        self.apply_to_classifications = apply_to_classifications

    @abstractmethod
    def calculate(self, deps: MetricDependencies, image: ByteTensor, mask: Optional[BoolTensor]) -> float:
        ...

    def calculate_for_object(self, deps: MetricDependencies, image: ByteTensor, mask: BoolTensor,
                             coordinates: FloatTensor) -> float:
        _coordinates = coordinates  # Only used in override logic for special case shortcuts
        return self.calculate(deps, image, mask)

    def calculate_for_image(self, deps: MetricDependencies, image: ByteTensor) -> float:
        return self.calculate(deps, image, None)


class OneObjectMetric(BaseAnalysis, metaclass=ABCMeta):
    """
    Simple metric that only depends on the object shape itself (& pixels via per-object embeddings)
    This is implicitly applied to all child classifications of an object.
    """

    @abstractmethod
    def calculate(self, deps: MetricDependencies, obj: ObjectMetadata) -> float:
        ...


class ObjectByFrameMetric(BaseAnalysis, metaclass=ABCMeta):
    """
    More complex single frame object metric where the result of each object depends on all other objects
    present in the single frame. The keys of the result should match the keys of 'objs'.
    """

    @abstractmethod
    def calculate(self, img_deps: MetricDependencies, obj_deps: Dict[str, MetricDependencies],
                  objs: Dict[str, ObjectMetadata]) -> Dict[str, float]:
        ...


class ImageObjectsMetric(BaseAnalysis, metaclass=ABCMeta):
    @abstractmethod
    def calculate(self, img_deps: MetricDependencies, obj_deps: Dict[str, MetricDependencies],
                  objs: Dict[str, ObjectMetadata]) -> float:
        ...


class TemporalOneImageMetric(TemporalBaseAnalysis, metaclass=ABCMeta):
    """
    Temporal variant of [OneImageMetric]
    """

    def __init__(self, ident: str, dependencies: Set[str], long_name: str, short_desc: str, long_desc: str,
                 prev_frame_count: int, next_frame_count: int,
                 apply_to_objects: bool = True, apply_to_classifications: bool = True) -> None:
        super().__init__(ident, dependencies, long_name, short_desc, long_desc, prev_frame_count, next_frame_count)
        self.apply_to_objects = apply_to_objects
        self.apply_to_classifications = apply_to_classifications

    @abstractmethod
    def calculate(self, deps: MetricDependencies, image: ByteTensor, mask: Optional[BoolTensor],
                  prev_frames: List[Tuple[MetricDependencies, ByteTensor, Optional[BoolTensor]]],
                  next_frames: List[Tuple[MetricDependencies, ByteTensor, Optional[BoolTensor]]]) -> float:
        ...


class TemporalOneObjectMetric(TemporalBaseAnalysis, metaclass=ABCMeta):
    """
    Temporal variant of [OneObjectMetric].
    """

    @abstractmethod
    def calculate(self, deps: MetricDependencies, obj: ObjectMetadata,
                  prev_frames: List[Tuple[MetricDependencies, ObjectMetadata]],
                  next_frames: List[Tuple[MetricDependencies, ObjectMetadata]]) -> float:
        ...


class TemporalObjectByFrameMetric(TemporalBaseAnalysis, metaclass=ABCMeta):
    @abstractmethod
    def calculate(self, img_deps: MetricDependencies, obj_deps: Dict[str, MetricDependencies],
                  objs: Dict[str, ObjectMetadata],
                  prev_frames: List[
                      Tuple[MetricDependencies, Dict[str, MetricDependencies], Dict[str, ObjectMetadata]]
                  ],
                  next_frames: List[
                      Tuple[MetricDependencies, Dict[str, MetricDependencies], Dict[str, ObjectMetadata]]
                  ],
                  ) -> Dict[str, float]:
        ...


class DerivedMetric(BaseAnalysis, metaclass=ABCMeta):
    """
    Simple metric that only depends on the pixels, if enabled the mask argument allows this metric to
    also apply to objects by only considering the subset of the image the object is present in.
    """

    def __init__(self, ident: str, dependencies: Set[str], long_name: str, short_desc: str, long_desc: str,
                 apply_to_images: bool = True,
                 apply_to_objects: bool = True,
                 apply_to_classifications: bool = True) -> None:
        super().__init__(ident, dependencies, long_name, short_desc, long_desc)
        self.apply_to_images = apply_to_images
        self.apply_to_objects = apply_to_objects
        self.apply_to_classifications = apply_to_classifications

    @abstractmethod
    def calculate(self, deps: MetricDependencies) -> float:
        ...