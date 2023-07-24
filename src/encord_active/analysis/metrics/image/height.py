from typing import Optional

import torch

from encord_active.analysis.base import BaseFrameAnnotationBatchInput
from encord_active.analysis.metric import MetricDependencies, OneImageMetric, ImageObjectOnlyOutputBatch, \
    ObjectOnlyBatchInput
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult, MetricBatchDependencies, \
    ImageBatchTensor, MaskBatchTensor, MetricBatchResult
from encord_active.analysis.util import image_height
from encord_active.analysis.util.torch import mask_to_box_extremes, batch_size


class HeightMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_height",
            dependencies=set(),
            long_name="Height",
            desc="Height in pixels.",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        if mask is None:
            return float(image_height(image))
        else:
            top_left, bottom_right = mask_to_box_extremes(mask)
            return bottom_right.y + 1 - top_left.y

    def calculate_batched(
        self,
        deps: MetricBatchDependencies,
        image: ImageBatchTensor,
        annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        img_height = image_height(image)
        img_batch = batch_size(image)

        objects = None
        if annotation is not None:
            # Returns shape=(batch, 4) where 4=(x1, y1, x2, y2)
            bounding_boxes = annotation.objects_bounding_boxes
            y1 = bounding_boxes[:, 1]
            y2 = bounding_boxes[:, 3]
            h = y2 - y1
            objects = h

        return ImageObjectOnlyOutputBatch(
            images=torch.full((img_batch,), img_height, dtype=torch.int64),
            objects=objects
        )

