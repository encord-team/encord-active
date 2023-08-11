from typing import Optional

import torch

from encord_active.analysis.metric import (
    ImageObjectOnlyOutputBatch,
    MetricDependencies,
    ObjectOnlyBatchInput,
    OneImageMetric,
)
from encord_active.analysis.types import (
    BoundingBoxTensor,
    ImageBatchTensor,
    ImageTensor,
    MaskTensor,
    MetricBatchDependencies,
    MetricResult,
)
from encord_active.analysis.util import image_height, image_width, mask_to_box_extremes
from encord_active.analysis.util.torch import batch_size
from encord_active.db.metrics import MetricType


class AspectRatioMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_aspect_ratio",
            long_name="Aspect Ratio",
            desc="",
            metric_type=MetricType.UFLOAT,
        )

    def calculate(
        self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor], bb: Optional[BoundingBoxTensor]
    ) -> MetricResult:
        if mask is None or bb is None:
            return float(image_width(image)) / float(image_height(image))
        else:
            x1, y1, x2, y2 = bb.type(torch.int32).tolist()
            return (x2 + 1 - x1) / (y2 + 1 - y1)

    def calculate_batched(
        self, deps: MetricBatchDependencies, image: ImageBatchTensor, annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        img_aspect_ratio = float(image_width(image)) / float(image_height(image))
        img_batch = batch_size(image)
        objects = None
        if objects is not None:
            # Returns shape=(batch, 4) where 4=(x1, y1, x2, y2)
            bounding_boxes = annotation.objects_bounding_boxes
            xy1 = bounding_boxes[:, :2]
            xy2 = bounding_boxes[:, 2:]
            wh = xy2 - xy1
            w = wh[:, 0]
            h = wh[:, 1]
            objects = w / h
        return ImageObjectOnlyOutputBatch(
            images=torch.full((img_batch,), img_aspect_ratio, dtype=torch.float32), objects=objects
        )
