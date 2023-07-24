from typing import Optional

import torch
from torchvision.ops import masks_to_boxes

from encord_active.analysis.base import BaseFrameAnnotationBatchInput
from encord_active.analysis.metric import MetricDependencies, OneImageMetric, ObjectOnlyBatchInput, \
    ImageObjectOnlyOutputBatch
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult, MetricBatchDependencies, \
    ImageBatchTensor, MaskBatchTensor, MetricBatchResult
from encord_active.analysis.util import image_height, image_width, mask_to_box_extremes
from encord_active.analysis.util.torch import batch_size


class AspectRatioMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_aspect_ratio",
            dependencies=set(),
            long_name="Aspect Ratio",
            desc="",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        if mask is None:
            return float(image_width(image)) / float(image_height(image))
        else:
            top_left, bottom_right = mask_to_box_extremes(mask)
            width = bottom_right.x + 1 - top_left.x
            height = bottom_right.y + 1 - top_left.y
            return width / height

    def calculate_batched(
        self,
        deps: MetricBatchDependencies,
        image: ImageBatchTensor,
        annotation: Optional[ObjectOnlyBatchInput]
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
            images=torch.full((img_batch,), img_aspect_ratio, dtype=torch.float32),
            objects=objects
        )
