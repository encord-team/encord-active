from encord_active.analysis.metric import OneObjectMetric
from encord_active.analysis.types import MetricResult, AnnotationMetadata, MetricDependencies


class DistanceToBorderMetric(OneObjectMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="distance-to-border",
            dependencies=set(),
            long_name="Distance in pixels to the border",
            desc="",
        )

    def calculate(self, annotation: AnnotationMetadata, deps: MetricDependencies) -> MetricResult:
        points = annotation.points
        xmin, ymin = points.min(0).values
        xmax, ymax = 1 - points.max(0).values
        return min(xmin, xmax, ymin, ymax)
