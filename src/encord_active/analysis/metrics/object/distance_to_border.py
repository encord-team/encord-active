from encord_active.analysis.metric import OneObjectMetric
from encord_active.analysis.types import MetricResult, ObjectMetadata


class DistanceToBorderMetric(OneObjectMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="distance-to-border",
            dependencies=set(),
            long_name="Distance in pixels to the border",
            desc="",
        )

    def calculate(self, obj: ObjectMetadata) -> MetricResult:
        points = obj.torch_points()
        xmin, ymin = points.min(0).values
        xmax, ymax = 1 - points.max(0).values
        return min(xmin, xmax, ymin, ymax)
