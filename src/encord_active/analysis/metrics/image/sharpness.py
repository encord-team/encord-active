from typing import Optional

from encord_active.analysis.metric import MetricDependencies, OneImageMetric
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult
from encord_active.analysis.util import laplacian2d
from encord_active.analysis.util.torch import mask_to_box_extremes


class SharpnessMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_sharpness",
            dependencies=set(),
            long_name="Sharpness",
            desc="",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        if mask is None:
            return laplacian2d(image).std() / 255
        else:
            top_left, bottom_right = mask_to_box_extremes(mask)
            image_crop = image[:, top_left.y : bottom_right.y + 1, top_left.x : bottom_right.x + 1]
            laplacian_box = laplacian2d(image_crop)
            laplacian_mask = laplacian_box[~mask[top_left.y : bottom_right.y + 1, top_left.x : bottom_right.x + 1]]
            return laplacian_mask.std() / 255
