from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Set, Tuple, Union

import torch

from encord_active.analysis.base import (
    BaseFrameAnnotationBatchInput,
    BaseFrameBatchInput,
    BaseFrameBatchOutput,
    BaseFrameInput,
    BaseFrameOutput,
    BaseMetric,
)
from encord_active.analysis.types import (
    AnnotationMetadata,
    BoundingBoxBatchTensor,
    BoundingBoxTensor,
    ImageBatchTensor,
    ImageIndexBatchTensor,
    ImageTensor,
    MaskBatchTensor,
    MaskTensor,
    MetricBatchDependencies,
    MetricBatchResult,
    MetricDependencies,
    MetricResult,
)
from encord_active.db.enums import AnnotationType
from encord_active.db.metrics import MetricType

MetricResultOptionalAnnotation = Union[MetricResult, Tuple[MetricResult, str]]


class BaseMetricWithAnnotationFilter(BaseMetric, metaclass=ABCMeta):
    """
    Simple metric that only depends on the pixels, if enabled the mask argument allows this metric to
    also apply to objects by only considering the subset of the image the object is present in.
    """

    def __init__(
        self,
        ident: str,
        long_name: str,
        desc: str,
        metric_type: MetricType,
        annotation_types: Optional[Set[AnnotationType]] = None,
    ) -> None:
        super().__init__(ident, long_name, desc, metric_type)
        self.annotation_types = annotation_types


class ObjectOnlyBatchInput(Protocol):
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


@dataclass
class ImageObjectOnlyOutputBatch:
    images: Optional[MetricBatchResult]
    """
    B x image results
    """
    objects: Optional[MetricBatchResult]
    """
    O x annotation results
    """


class OneImageMetric(BaseMetricWithAnnotationFilter, metaclass=ABCMeta):
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
        image_raw = self.calculate(frame.image_deps, frame.image, None, None)
        if isinstance(image_raw, tuple):
            image, image_comment = image_raw
        else:
            image = image_raw
            image_comment = None
        annotations = {}
        annotation_comments = {}
        for annotation_hash, annotation in frame.annotations.items():
            if self.annotation_types is not None and annotation.annotation_type not in self.annotation_types:
                continue
            annotation_deps = frame.annotations_deps[annotation_hash]
            if annotation.annotation_type != AnnotationType.CLASSIFICATION:
                annotation_raw = self.calculate(
                    deps=annotation_deps,
                    image=frame.image,
                    mask=annotation.mask,
                    bb=annotation.bounding_box,
                )
                if isinstance(annotation_raw, tuple):
                    annotation_metric, annotation_comment = annotation_raw
                else:
                    annotation_metric = annotation_raw
                    annotation_comment = None
                annotations[annotation_hash] = annotation_metric
                if annotation_comment is not None:
                    annotation_comments[annotation_hash] = annotation_comment
            else:
                annotations[annotation_hash] = image
                if image_comment is not None:
                    annotation_comments[annotation_hash] = image_comment
        return BaseFrameOutput(
            image=image, annotations=annotations, image_comment=image_comment, annotation_comments=annotation_comments
        )

    @abstractmethod
    def calculate(
        self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor], bb: Optional[BoundingBoxTensor]
    ) -> MetricResultOptionalAnnotation:
        ...

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        """
        Base implementation of batched metric calculation.
        """
        res = self.calculate_batched(
            frame.images_deps,
            frame.images,
            frame.annotations,
        )
        classifications = None
        if frame.annotations is not None and res.images is not None:
            classifications = torch.index_select(res.images, 0, frame.annotations.classifications_image_indices)
        return BaseFrameBatchOutput(
            images=res.images,
            objects=res.objects,
            classifications=classifications,
        )

    @abstractmethod
    def calculate_batched(
        self, deps: MetricBatchDependencies, image: ImageBatchTensor, annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        ...


class OneObjectMetric(BaseMetricWithAnnotationFilter, metaclass=ABCMeta):
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
        annotation_comments = {}
        for annotation_hash, annotation in frame.annotations.items():
            if self.annotation_types is not None and annotation.annotation_type not in self.annotation_types:
                continue
            annotation_deps = frame.annotations_deps[annotation_hash]
            annotation_raw = self.calculate(
                annotation=annotation,
                deps=annotation_deps,
            )
            if isinstance(annotation_raw, tuple):
                annotation_metric, annotation_comment = annotation_raw
            else:
                annotation_metric = annotation_raw
                annotation_comment = None
            annotations[annotation_hash] = annotation_metric
            if annotation_comment is not None:
                annotation_comments[annotation_hash] = annotation_comment
        return BaseFrameOutput(
            image=None,
            annotations=annotations,
            image_comment=None,
            annotation_comments=annotation_comments,
        )

    @abstractmethod
    def calculate(self, annotation: AnnotationMetadata, deps: MetricDependencies) -> MetricResultOptionalAnnotation:
        ...

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        """
        Base implementation of batched metric calculation.
        """
        return self.calculate_batched(
            frame.images_deps,
            frame.images,
            frame.annotations,
        )

    @abstractmethod
    def calculate_batched(
        self,
        deps: MetricBatchDependencies,
        image: ImageBatchTensor,
        annotation: Optional[BaseFrameAnnotationBatchInput],
    ) -> BaseFrameBatchOutput:
        ...


class ObjectByFrameMetric(BaseMetric, metaclass=ABCMeta):
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
        if len(frame.annotations) > 0:
            annotations_raw = self.calculate(
                frame.annotations,
                frame.annotations_deps,
            )
        else:
            annotations_raw = {}
        return BaseFrameOutput(
            image=None,
            image_comment=None,
            annotations={k: v[0] if isinstance(v, tuple) else v for k, v in annotations_raw.items()},
            annotation_comments={k: v[1] for k, v in annotations_raw.items() if isinstance(v, tuple)},
        )

    @abstractmethod
    def calculate(
        self,
        annotations: Dict[str, AnnotationMetadata],
        annotation_deps: dict[str, MetricDependencies],
    ) -> Dict[str, MetricResultOptionalAnnotation]:
        ...

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        raise ValueError("Not implemented")

    @abstractmethod
    def calculate_batched(
        self,
        masks: MaskBatchTensor,
    ) -> MetricBatchResult:
        ...


class ImageObjectsMetric(BaseMetric, metaclass=ABCMeta):
    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        image_raw = self.calculate(
            frame.image,
            frame.image_deps,
            frame.annotations,
            frame.annotations_deps,
        )
        return BaseFrameOutput(
            image=image_raw[0] if isinstance(image_raw, tuple) else image_raw,
            image_comment=image_raw[1] if isinstance(image_raw, tuple) else None,
            annotations={},
            annotation_comments={},
        )

    @abstractmethod
    def calculate(
        self,
        image: ImageTensor,
        image_deps: MetricDependencies,
        # key is object_hash | classification_hash
        annotations: Dict[str, AnnotationMetadata],
        annotations_deps: Dict[str, MetricDependencies],
    ) -> MetricResultOptionalAnnotation:
        """
        TODO: This is currently only used by object count which doesn't require all these arguments.
        """
        ...

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        raise ValueError("Batch not yet implemented")


class TemporalOneImageMetric(BaseMetricWithAnnotationFilter, metaclass=ABCMeta):
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
            return BaseFrameOutput(
                image=self.default_value(),
                annotations={annotation_hash: self.default_value() for annotation_hash in frame.annotations.keys()},
                image_comment=None,
                annotation_comments={},
            )
        image_raw = self.calculate(
            deps=frame.image_deps,
            image=frame.image,
            mask=None,
            prev_image=prev_frame.image,
            prev_mask=None,
            next_image=next_frame.image,
            next_mask=None,
        )
        image = image_raw[0] if isinstance(image_raw, tuple) else image_raw
        image_comment = image_raw[1] if isinstance(image_raw, tuple) else None
        annotations = {}
        annotation_comments = {}
        for annotation_hash, annotation in frame.annotations.items():
            if self.annotation_types is not None and annotation.annotation_type not in self.annotation_types:
                continue
            annotation_deps = frame.annotations_deps[annotation_hash]
            if annotation.annotation_type != AnnotationType.CLASSIFICATION:
                prev_annotation = None if prev_frame is None else prev_frame.annotations.get(annotation_hash, None)
                next_annotation = None if next_frame is None else next_frame.annotations.get(annotation_hash, None)
                annotation_raw = self.calculate(
                    deps=annotation_deps,
                    image=frame.image,
                    mask=annotation.mask,
                    prev_image=prev_frame.image,
                    prev_mask=None if prev_annotation is None else prev_annotation.mask,
                    next_image=next_frame.image,
                    next_mask=None if next_annotation is None else next_annotation.mask,
                )
                annotations[annotation_hash] = (
                    annotation_raw[0] if isinstance(annotation_raw, tuple) else annotation_raw
                )
                if isinstance(annotation_raw, tuple):
                    annotation_comment = annotation_raw[1]
                    if annotation_comment is not None:
                        annotation_comments[annotation_hash] = annotation_comment
            else:
                annotations[annotation_hash] = image
                if image_comment is not None:
                    annotation_comments[annotation_hash] = image_comment
        return BaseFrameOutput(
            image=image,
            image_comment=image_comment,
            annotations=annotations,
            annotation_comments=annotation_comments,
        )

    @abstractmethod
    def default_value(self) -> MetricResult:
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
    ) -> MetricResultOptionalAnnotation:
        ...

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        raise ValueError("Batch not yet implemented")


class TemporalOneObjectMetric(BaseMetricWithAnnotationFilter, metaclass=ABCMeta):
    """
    Temporal variant of [OneObjectMetric].
    """

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        annotations = {}
        annotation_comments = {}
        for annotation_hash, annotation in frame.annotations.items():
            if self.annotation_types is not None and annotation.annotation_type not in self.annotation_types:
                continue
            annotation_deps = frame.annotations_deps[annotation_hash]
            annotation_raw = self.calculate(
                annotation=annotation,
                deps=annotation_deps,
                prev_annotation=None if prev_frame is None else prev_frame.annotations.get(annotation_hash, None),
                next_annotation=None if next_frame is None else next_frame.annotations.get(annotation_hash, None),
            )
            annotations[annotation_hash] = annotation_raw[0] if isinstance(annotation_raw, tuple) else annotation_raw
            if isinstance(annotation_raw, tuple):
                annotation_comment = annotation_raw[1]
                if annotation_comment is not None:
                    annotation_comments[annotation_hash] = annotation_comment
        return BaseFrameOutput(
            image=None, image_comment=None, annotations=annotations, annotation_comments=annotation_comments
        )

    @abstractmethod
    def calculate(
        self,
        annotation: AnnotationMetadata,
        deps: MetricDependencies,
        prev_annotation: Optional[AnnotationMetadata],
        next_annotation: Optional[AnnotationMetadata],
    ) -> MetricResultOptionalAnnotation:
        ...

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        raise ValueError("Batch not yet implemented")


class TemporalObjectByFrameMetric(BaseMetric, metaclass=ABCMeta):
    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        annotations_raw = self.calculate(
            annotations=frame.annotations,
            annotation_deps=frame.annotations_deps,
            prev_annotations=None if prev_frame is None else prev_frame.annotations,
            next_annotations=None if next_frame is None else next_frame.annotations,
        )
        return BaseFrameOutput(
            image=None,
            image_comment=None,
            annotations={k: v[0] if isinstance(v, tuple) else v for k, v in annotations_raw.items()},
            annotation_comments={k: v[1] for k, v in annotations_raw.items() if isinstance(v, tuple)},
        )

    @abstractmethod
    def calculate(
        self,
        annotations: Dict[str, AnnotationMetadata],
        annotation_deps: dict[str, MetricDependencies],
        prev_annotations: Optional[Dict[str, AnnotationMetadata]],
        next_annotations: Optional[Dict[str, AnnotationMetadata]],
    ) -> Dict[str, MetricResultOptionalAnnotation]:
        ...

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        raise ValueError("Batch not yet implemented")


class DerivedMetric(BaseMetric, metaclass=ABCMeta):
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
        raise AttributeError("Derived uses separate interface, this function should NEVER be called!")

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        raise AttributeError("Derived uses separate interface, this function should NEVER be called!")

    @abstractmethod
    def calculate(
        self, deps: MetricDependencies, annotation: Optional[AnnotationMetadata]
    ) -> MetricResultOptionalAnnotation:
        ...


class RandomSamplingQuery:
    """
    Randomly select 'limit' number of entries from the analysis scope.
    """

    def __init__(self, ident: str, limit: int = 100, same_feature_hash: bool = False) -> None:
        self.ident = ident
        self.limit = limit
        self.same_feature_hash = same_feature_hash

    # FIXME: add support for 'filters' (constraint the randomness depending on sql-like filters)??
