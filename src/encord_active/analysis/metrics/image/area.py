from typing import Optional

import torch

from encord_active.analysis.base import BaseFrameAnnotationBatchInput
from encord_active.analysis.metric import MetricDependencies, OneImageMetric, ImageObjectOnlyOutputBatch, \
    ObjectOnlyBatchInput
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult, MetricBatchDependencies, \
    ImageBatchTensor, MaskBatchTensor, MetricBatchResult
from encord_active.analysis.util import image_height, image_width
from encord_active.analysis.util.torch import batch_size


class AreaMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_area",
            dependencies=set(),
            long_name="Area",
            desc="Area in pixels",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        if mask is None:
            return float(image_width(image)) * float(image_height(image))
        else:
            return float(torch.sum(mask.long()).item())

    def calculate_batched(
        self,
        deps: MetricBatchDependencies,
        image: ImageBatchTensor,
        annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        image_area = image_width(image) * image_height(image)
        image_batch = batch_size(image)
        objects = None
        if annotation is not None:
            objects = torch.sum(annotation.objects_masks.long(), dim=(1, 2), dtype=torch.int64)
        return ImageObjectOnlyOutputBatch(
            images=torch.full((image_batch,), image_area, dtype=torch.int64),
            objects=objects
        )


class AreaRelativeMetric(OneImageMetric):  # FIXME: OneObjectMetric
    def __init__(self) -> None:
        super().__init__(
            ident="metric_area_relative",
            dependencies=set(),
            long_name="Area Relative",
            desc="Area in pixels relative to image area",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        if mask is None:
            return float(image_width(image)) * float(image_height(image))
        else:
            obj_area = float(deps["metric_area"])
            area = float(image_width(image)) * float(image_height(image))
            return obj_area / area

    def calculate_batched(
        self,
        deps: MetricBatchDependencies,
        image: ImageBatchTensor,
        annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        if annotation is None:
            return None
        else:
            area = float(image_width(image)) * float(image_height(image))
            obj_area = annotation.annotations_deps["metric_area"]
            return obj_area.type(dtype=torch.float32) / area
        return ImageObjectOnlyOutputBatch(
            images=None,
            objects=None
        )

