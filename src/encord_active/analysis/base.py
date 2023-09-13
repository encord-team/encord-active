from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Set

from encord_active.analysis.types import (
    AnnotationMetadata,
    BoundingBoxBatchTensor,
    FeatureHashBatchTensor,
    ImageBatchTensor,
    ImageIndexBatchTensor,
    ImageTensor,
    MaskBatchTensor,
    MetricBatchDependencies,
    MetricBatchResult,
    MetricDependencies,
    MetricResult,
)
from encord_active.db.enums import DataType
from encord_active.db.metrics import MetricType


@dataclass(frozen=True)  # FIXME: deprecate for new version
class BaseFrameInput:
    image: ImageTensor
    image_deps: MetricDependencies
    # key is object_hash | classification_hash
    annotations: Dict[str, AnnotationMetadata]
    annotations_deps: Dict[str, MetricDependencies]
    # hash collision, if set the object or classification hash uniqueness constraint is
    # not held for this frame
    hash_collision: Set[str]
    data_type: DataType


@dataclass
class BaseFrameOutput:
    image: Optional[MetricResult]
    annotations: Dict[str, MetricResult]


@dataclass
class BaseFrameAnnotationBatchInput:
    """
    Dictionary to B x dependencies
    """

    objects_masks: MaskBatchTensor
    """
    O x Masks (O = SUM{i.annotation_count * B}
    """
    objects_bounding_boxes: BoundingBoxBatchTensor
    """"
    O x Bounding Boxes
    """
    objects_deps: MetricBatchDependencies
    """
    dependency map to Ox<value> tensors
    """
    objects_image_indices: ImageIndexBatchTensor
    """
    O x int32 index to the associated image.
    """
    objects_feature_hash: FeatureHashBatchTensor
    """
    O x int64 converted feature hash values
    """
    classifications_deps: MetricBatchDependencies
    """
    dependency map to Cx<value> tensors
    """
    classifications_image_indices: ImageIndexBatchTensor
    """
    C x int32 index to the associated image.
    """
    classifications_feature_hash: FeatureHashBatchTensor
    """
    C x int64 converted feature hash values
    """


@dataclass
class BaseFrameBatchInput:
    images: ImageBatchTensor
    """
    B x image tensors
    """
    images_deps: MetricBatchDependencies
    """
    Dictionary to B x dependencies
    """
    annotations: Optional[BaseFrameAnnotationBatchInput]
    """
    All annotation batches associated with this image batch.
    Values are split into classifications & objects (objects have masks + aabb)
    """


@dataclass
class BaseFrameBatchOutput:
    images: Optional[MetricBatchResult]
    """
    B x image results
    """
    objects: Optional[MetricBatchResult]
    """
    O x annotation results
    """
    classifications: Optional[MetricBatchResult]
    """
    C x classification results
    """


class BaseEvaluation(ABC):
    """
    Base evaluations are the most abstract definitions of computations that need to happen.
    They can be both:

    - Quality Metrics that needs to be stored (The `BaseAnalysis` subclass)
    - Computations the need to happen for Quality Metrics to be able to share computations.
      For example, Embeddings, image transforms, etc.

    """

    def __init__(self, ident: str) -> None:
        self.ident = ident

    @abstractmethod
    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        """
        Base implementation of the raw_calculate method, this api should
        be considered unstable.
        """
        ...

    @abstractmethod
    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        """
        Base implementation of batched metric calculation.
        """


class BaseAnalysis(BaseEvaluation, metaclass=ABCMeta):
    """
    The `BaseAnalysis` is all the metrics that needs to be stored in the
    database.
    """

    def __init__(self, ident: str, long_name: str, desc: str) -> None:
        super().__init__(ident)
        self.long_name = long_name
        self.desc = desc


class BaseMetric(BaseAnalysis, metaclass=ABCMeta):
    def __init__(self, ident: str, long_name: str, desc: str, metric_type: MetricType) -> None:
        super().__init__(ident, long_name, desc)
        self.metric_type = metric_type
