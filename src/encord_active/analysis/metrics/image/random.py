import torch

from encord_active.analysis.metric import MetricDependencies, OneImageMetric
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult


class RandomMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_random",
            dependencies=set(),
            long_name="Random Value",
            desc="Assigns random float value in the range [0; 1].",
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: MaskTensor | None) -> MetricResult:
        return torch.rand(1)
