from typing import Optional

import torch

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
            ident="metric_polygon_similarity",
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
            return torch.nn.CosineSimilarity(dim=0)(deps["weighted_hu_embedding"], deps["embedding_hu"])
        return None
