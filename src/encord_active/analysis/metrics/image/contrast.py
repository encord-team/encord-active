import torch

from encord_active.analysis.metric import MetricDependencies, OneImageMetric
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult


class ContrastMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="contrast",
            dependencies={"brightness", "area"},
            long_name="Contrast",
            desc="",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: MaskTensor | None) -> MetricResult:
        if mask is None:
            return torch.std(image).item() / 255.0
        else:
            # Masked standard deviation
            mask_mean = float(deps["img-brightness"])
            mask_count = float(deps["img-area"])

            mask_mean_delta = torch.masked_fill(image.float() - mask_mean, ~mask, 0)
            mask_mean_delta_sq_sum = torch.sum(mask_mean_delta**2)

            return torch.sqrt(mask_mean_delta_sq_sum / mask_count) / 255.0
