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
            return torch.mean(image, dtype=torch.float) / 255.0
        else:
            mask_count = float(deps["metric_area"])
            image_reduced = torch.round(torch.mean(image, dim=0, dtype=torch.float32))
            mask_total = torch.sum(torch.masked_fill(image_reduced, ~mask, 0), dtype=torch.float32).item()
            return float(mask_total) / (255.0 * mask_count)
