from typing import Optional

from encord_active.analysis.metric import TemporalOneObjectMetric
from encord_active.analysis.types import (
    AnnotationMetadata,
    MetricDependencies,
    MetricResult,
)
from encord_active.db.metrics import MetricType


class TemporalShapeChange(TemporalOneObjectMetric):
    def __init__(self):
        super().__init__(
            ident="metric_label_poly_similarity",
            long_name="Shape Outlier",
            desc="",
            metric_type=MetricType.NORMAL,
        )

    def calculate(
        self,
        annotation: AnnotationMetadata,
        deps: MetricDependencies,
        prev_annotation: Optional[AnnotationMetadata],
        next_annotation: Optional[AnnotationMetadata],
    ) -> MetricResult:
        if prev_annotation is None or next_annotation is None:
            return 0.0  # no obvious outlier
        # FIXME: Please actually implement this.
        return None
