import torch
from typing import Optional
from encord_active.analysis.metric import MetricDependencies, OneImageMetric
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult


class BrightnessMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_brightness",
            dependencies={"area"},
            long_name="Brightness",
            desc="",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        if mask is None:
            return torch.mean(image).item() / 255
        else:
            mask_count = float(deps["img-area"])
            mask_total = torch.sum(torch.masked_fill(image.long(), ~mask, 0)).item()
            return float(mask_total) / mask_count
