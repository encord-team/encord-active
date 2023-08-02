from typing import Optional, cast

from encord_active.analysis.metric import DerivedMetric
from encord_active.analysis.types import (
    AnnotationMetadata,
    MetricDependencies,
    MetricResult,
    NearestNeighbors,
)
from encord_active.db.metrics import MetricType

"""
FIXME: we don't want rank metrics - some summary values lose information by making ranked.
class ImageSingularity(OneImageMetric):
    pass
"""


class ImageUniqueness(DerivedMetric):
    def __init__(self):
        super().__init__(
            ident="metric_image_uniqueness",
            long_name="Image Uniqueness",
            desc="How unique an image is in the dataset",
            metric_type=MetricType.NORMAL,
        )

    def calculate(self, deps: MetricDependencies, annotation: Optional[AnnotationMetadata]) -> MetricResult:
        if annotation is not None:
            return None
        # FIXME: validate normalisation of this value (is 1.0 distance clamp reasonable??)
        nearest_neighbours: NearestNeighbors = cast(NearestNeighbors, deps["derived_clip_nearest"])
        if len(nearest_neighbours.similarities) == 0:
            return 1.0
        else:
            # Clamp large distances to (unique)
            # FIXME: filter out nearby frames / same video?
            return min(nearest_neighbours.similarities[0], 1.0)