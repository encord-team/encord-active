from typing import Optional

import torch

from encord_active.analysis.metric import (
    ImageObjectOnlyOutputBatch,
    MetricDependencies,
    ObjectOnlyBatchInput,
    OneImageMetric,
)
from encord_active.analysis.types import (
    ImageBatchTensor,
    ImageTensor,
    MaskTensor,
    MetricBatchDependencies,
    MetricResult,
)
from encord_active.analysis.util import laplacian2d
from encord_active.analysis.util.torch import mask_to_box_extremes
from encord_active.db.metrics import MetricType


class SharpnessMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_sharpness",
            long_name="Sharpness",
            desc="",
            metric_type=MetricType.NORMAL,
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        # Max range of laplacian kernel = [-4, 4] * 255 = [-1020, 1020]
        # Max value of variance = (2.0 * 1020.0 * 1020.0) = 2080800.
        # FIXME: grayscale VS rgb??? (WHICH SHOULD WE USE!!!!!)
        grayscale = torch.mean(image, dim=-3, dtype=torch.float32).type(torch.uint8).unsqueeze(0)
        if mask is None:
            # FIXME: variance VS stddev (which gives better summaries for normalisation)
            return torch.var(laplacian2d(grayscale)) / 2080800.0
        else:
            top_left, bottom_right = mask_to_box_extremes(mask)
            image_crop = grayscale[:, top_left.y : bottom_right.y + 1, top_left.x : bottom_right.x + 1]
            laplacian_box = laplacian2d(image_crop).squeeze(0)
            laplacian_mask = laplacian_box[~mask[top_left.y : bottom_right.y + 1, top_left.x : bottom_right.x + 1]]
            return laplacian_mask.var() / 2080800.0

    def calculate_batched(
        self, deps: MetricBatchDependencies, image: ImageBatchTensor, annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        # max-laplacian = [4x255
        grayscale = deps["ephemeral_grayscale_image"]
        laplacian = laplacian2d(grayscale)
        raise ValueError("Not yet implemented properly")
