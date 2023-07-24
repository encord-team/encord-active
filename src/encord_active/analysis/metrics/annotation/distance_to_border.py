from typing import Optional

from encord_active.analysis.base import BaseFrameAnnotationBatchInput, BaseFrameBatchOutput
from encord_active.analysis.metric import OneObjectMetric
from encord_active.analysis.types import MetricResult, AnnotationMetadata, MetricDependencies, MetricBatchDependencies, \
    ImageBatchTensor


class DistanceToBorderMetric(OneObjectMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_label_border_closeness",
            dependencies=set(),
            long_name="Distance in pixels to the border",
            desc="",
        )

    def calculate(self, annotation: AnnotationMetadata, deps: MetricDependencies) -> MetricResult:
        points = annotation.points
        mask = annotation.mask
        if points is not None:
            x_min, y_min = points.min(0).values.cpu()
            x_max, y_max = 1 - points.max(0).values.cpu()
            return min(x_min, x_max, y_min, y_max)
        elif mask is None:
            # Classification (distance = 0) FIXME: correct fallback or should this be NULL?!
            return 0
        else:
            raise ValueError(f"Border closeness not supported for bitmasks yet!")

    def calculate_batched(
        self,
        deps: MetricBatchDependencies,
        image: ImageBatchTensor,
        annotation: Optional[BaseFrameAnnotationBatchInput]
    ) -> BaseFrameBatchOutput:
        raise ValueError(f"Not yet implemented for batching")

