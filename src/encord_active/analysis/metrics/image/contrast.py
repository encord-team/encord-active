from typing import Optional

import torch

from encord_active.analysis.metric import MetricDependencies, OneImageMetric
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult


class ContrastMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_contrast",
            dependencies={"brightness", "area"},
            long_name="Contrast",
            desc="",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        image_reduced = torch.round(torch.mean(image, dim=0, dtype=torch.float32))
        if mask is None:
            return torch.std(image_reduced) / 255.0
        else:
            # Masked standard deviation
            mask_mean = float(deps["metric_brightness"])
            mask_count = float(deps["metric_area"])

            mask_mean_delta = torch.masked_fill(image_reduced - mask_mean, ~mask, 0)
            mask_mean_delta_sq_sum = torch.sum(mask_mean_delta**2, dtype=torch.float32)

            return torch.sqrt(mask_mean_delta_sq_sum / mask_count) / 255.0
