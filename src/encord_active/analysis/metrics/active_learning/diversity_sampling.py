from typing import Optional, cast

from encord_active.analysis.metric import DerivedMetric
from encord_active.analysis.types import (
    AnnotationMetadata,
    MetricDependencies,
    MetricResult,
    NearestNeighbors,
)
from encord_active.db.metrics import MetricType


class ImageDiversity(DerivedMetric):
    def __init__(self):
        super().__init__(
            ident="metric_image_diversity",
            long_name="Image Diversity",
            desc="Average distances to the neighborhood samples, which is a proxy for how"
            " diverse the image compared to its neighbors",
            metric_type=MetricType.NORMAL,
        )

    def calculate(self, deps: MetricDependencies, annotation: Optional[AnnotationMetadata]) -> MetricResult:
        if annotation is not None:
            return None
        nearest_neighbours: NearestNeighbors = cast(NearestNeighbors, deps["derived_clip_nearest_cosine"])

        if len(nearest_neighbours.similarities) == 0:
            return 1.0
        else:
            return min(sum(nearest_neighbours.similarities) / len(nearest_neighbours.similarities), 1.0)
