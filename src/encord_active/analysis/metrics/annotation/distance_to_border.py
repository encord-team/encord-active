from typing import Optional

import torch

from encord_active.analysis.base import (
    BaseFrameAnnotationBatchInput,
    BaseFrameBatchOutput,
)
from encord_active.analysis.metric import OneObjectMetric
from encord_active.analysis.types import (
    AnnotationMetadata,
    ImageBatchTensor,
    MetricBatchDependencies,
    MetricDependencies,
    MetricResult,
)
from encord_active.analysis.util import image_height, image_width
from encord_active.db.metrics import MetricType


class DistanceToBorderMetric(OneObjectMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_distance_to_border_relative",
            long_name="Closest distance to border relatively",
            desc="Returns the relative closest distance to any border.",
            metric_type=MetricType.NORMAL,
        )

    def calculate(self, annotation: AnnotationMetadata, deps: MetricDependencies) -> MetricResult:
        bb = annotation.bounding_box
        mask = annotation.mask
        # FIXME: change to use bounding box!! (should give same answer).
        if mask is not None and bb is not None:
            w = image_width(mask)
            h = image_height(mask)
            x1, y1, x2, y2 = bb.type(torch.int32).tolist()
            dx1 = float(x1 / w) / (w / 2)
            dx2 = float((w - (x2 + 1)) / w) / (w / 2)
            dy1 = float(y1 / h) / (h / 2)
            dy2 = float((h - (y2 + 1)) / h) / (h / 2)
            return min(dx1, dx2, dy1, dy2)
        else:
            # Classification
            return 0.0

    def calculate_batched(
        self,
        deps: MetricBatchDependencies,
        image: ImageBatchTensor,
        annotation: Optional[BaseFrameAnnotationBatchInput],
    ) -> BaseFrameBatchOutput:
        raise ValueError("Not yet implemented for batching")
