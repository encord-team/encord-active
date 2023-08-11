import math
from typing import Optional, cast

import torch

from encord_active.analysis.metric import (
    ImageObjectOnlyOutputBatch,
    MetricDependencies,
    ObjectOnlyBatchInput,
    OneImageMetric,
)
from encord_active.analysis.types import (
    BoundingBoxTensor,
    ImageBatchTensor,
    ImageTensor,
    MaskTensor,
    MetricBatchDependencies,
    MetricResult,
)
from encord_active.analysis.util import laplacian2d
from encord_active.db.metrics import MetricType


class SharpnessMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_sharpness",
            long_name="Sharpness",
            desc="",
            metric_type=MetricType.NORMAL,
        )

    def calculate(
        self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor], bb: Optional[BoundingBoxTensor]
    ) -> MetricResult:
        # Max range of laplacian kernel = [-4, 4] * 255 = [-1020, 1020]
        # Max value of variance = (2.0 * 1020.0 * 1020.0) = 2080800.
        laplacian = cast(torch.Tensor, deps["ephemeral_laplacian_image"])
        return math.sqrt(torch.var(laplacian).cpu().item() / 2080800.0)

    def calculate_batched(
        self, deps: MetricBatchDependencies, image: ImageBatchTensor, annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        # max-laplacian = [4x255
        grayscale = deps["ephemeral_grayscale_image"]
        laplacian = laplacian2d(grayscale)
        raise ValueError("Not yet implemented properly")
