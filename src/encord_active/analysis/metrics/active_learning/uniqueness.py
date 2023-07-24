from typing import Optional

from encord_active.analysis.metric import OneImageMetric, DerivedMetric
from encord_active.analysis.types import MetricDependencies, AnnotationMetadata, MetricResult

"""
FIXME: we don't want rank metrics - some summary values lose information by making ranked.
class ImageSingularity(OneImageMetric):
    pass
"""


class ImageUniqueness(DerivedMetric):

    def __init__(self):
        super().__init__(
            ident="metric_image_uniqueness",
            dependencies=set(),
            long_name="Image Uniqueness",
            desc="How unique an image is in the dataset",
        )

    def calculate(self, deps: MetricDependencies, annotation: Optional[AnnotationMetadata]) -> MetricResult:

        return None

