from encord_active.analysis.metric import MetricDependencies, OneImageMetric
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult
from encord_active.analysis.util import image_width
from encord_active.analysis.util.torch import mask_to_box_extremes


class WidthMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="width",
            dependencies=set(),
            long_name="Width",
            desc="Width in pixels.",
            apply_to_objects=False,
            apply_to_classifications=False,
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: MaskTensor | None) -> MetricResult:
        if mask is None:
            return float(image_width(image))
        else:
            top_left, bottom_right = mask_to_box_extremes(mask)
            return bottom_right.x + 1 - top_left.x
