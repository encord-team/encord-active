from typing import Optional

import torch

from encord_active.analysis.metric import (
    ImageObjectOnlyOutputBatch,
    MetricDependencies,
    ObjectOnlyBatchInput,
    OneImageMetric,
)
from encord_active.analysis.types import (
    ImageBatchTensor,
    ImageTensor,
    MaskTensor,
    MetricBatchDependencies,
    MetricResult,
)
from encord_active.analysis.util import image_width
from encord_active.analysis.util.torch import batch_size, mask_to_box_extremes
from encord_active.db.metrics import MetricType


class WidthMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_width",
            long_name="Width",
            desc="Width in pixels.",
            metric_type=MetricType.UINT,
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        if mask is None:
            return float(image_width(image))
        else:
            top_left, bottom_right = mask_to_box_extremes(mask)
            return bottom_right.x + 1 - top_left.x

    def calculate_batched(
        self, deps: MetricBatchDependencies, image: ImageBatchTensor, annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        img_width = image_width(image)
        img_batch = batch_size(image)

        objects = None
        if annotation is not None:
            # Returns shape=(batch, 4) where 4=(x1, y1, x2, y2)
            bounding_boxes = annotation.objects_bounding_boxes
            x1 = bounding_boxes[:, 0]
            x2 = bounding_boxes[:, 2]
            w = x2 - x1
            objects = w

        return ImageObjectOnlyOutputBatch(
            images=torch.full((img_batch,), img_width, dtype=torch.int64),
            objects=objects,
        )
