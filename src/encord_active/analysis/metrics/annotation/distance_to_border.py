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
            ident="metric_border_relative",  # FIXME: rename ident to something better??
            long_name="Distance in pixels to the border",
            desc="",
            metric_type=MetricType.NORMAL,
        )

    def calculate(self, annotation: AnnotationMetadata, deps: MetricDependencies) -> MetricResult:
        # FIXME: make relative?? (NORMAL metrics are better)
        points = annotation.points
        mask = annotation.mask
        # FIXME: change to use bounding box!! (should give same answer).
        if points is not None and mask is not None:
            w = image_width(mask)
            h = image_height(mask)
            sf = torch.tensor([[w, h]])
            points_sf = points / sf
            x_min, y_min = points_sf.min(0).values.cpu()
            x_max, y_max = 1.0 - points_sf.max(0).values.cpu()
            return min(x_min, x_max, y_min, y_max)
        elif mask is None and points is None:
            # FIXME: does not handle bitmask correctly!?
            # Classification (distance = 0) FIXME: correct fallback or should this be NULL?!
            return 0.0
        else:
            # FIXME: implement (use bounding box).
            raise ValueError("Border closeness not supported for bitmasks yet!")

    def calculate_batched(
        self,
        deps: MetricBatchDependencies,
        image: ImageBatchTensor,
        annotation: Optional[BaseFrameAnnotationBatchInput],
    ) -> BaseFrameBatchOutput:
        raise ValueError("Not yet implemented for batching")
