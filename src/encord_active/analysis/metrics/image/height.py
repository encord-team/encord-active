from encord_active.analysis.metric import MetricDependencies, OneImageMetric
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult
from encord_active.analysis.util import image_height
from encord_active.analysis.util.torch import mask_to_box_extremes


class HeightMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_height",
            dependencies=set(),
            long_name="Height",
            desc="Height in pixels.",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: MaskTensor | None) -> MetricResult:
        if mask is None:
            return float(image_height(image))
        else:
            top_left, bottom_right = mask_to_box_extremes(mask)
            return bottom_right.y + 1 - top_left.y
