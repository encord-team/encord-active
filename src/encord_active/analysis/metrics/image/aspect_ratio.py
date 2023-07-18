from typing import Optional
from encord_active.analysis.metric import MetricDependencies, OneImageMetric
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult
from encord_active.analysis.util import image_height, image_width, mask_to_box_extremes


class AspectRatioMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_aspect_ratio",
            dependencies=set(),
            long_name="Aspect Ratio",
            desc="",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        if mask is None:
            return float(image_width(image)) / float(image_height(image))
        else:
            top_left, bottom_right = mask_to_box_extremes(mask)
            width = bottom_right.x + 1 - top_left.x
            height = bottom_right.y + 1 - top_left.y
            return width / height
