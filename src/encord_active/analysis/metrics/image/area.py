from typing import Optional

import torch

from encord_active.analysis.metric import MetricDependencies, OneImageMetric
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult
from encord_active.analysis.util import image_height, image_width


class AreaMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_area",
            dependencies=set(),
            long_name="Area",
            desc="Area in pixels",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        if mask is None:
            return float(image_width(image)) * float(image_height(image))
        else:
            return float(torch.sum(mask.long()).item())
