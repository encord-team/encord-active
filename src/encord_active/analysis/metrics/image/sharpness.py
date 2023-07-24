from typing import Optional

import torch

from encord_active.analysis.base import BaseFrameAnnotationBatchInput
from encord_active.analysis.metric import MetricDependencies, OneImageMetric, ImageObjectOnlyOutputBatch, \
    ObjectOnlyBatchInput
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult, MetricBatchDependencies, \
    ImageBatchTensor, MaskBatchTensor, MetricBatchResult
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
        grayscale = torch.mean(image, dim=-3, dtype=torch.float32).type(torch.uint8).unsqueeze(0)
        if mask is None:
            return laplacian2d(grayscale).std() / 255
        else:
            top_left, bottom_right = mask_to_box_extremes(mask)
            image_crop = grayscale[:, top_left.y : bottom_right.y + 1, top_left.x : bottom_right.x + 1]
            laplacian_box = laplacian2d(image_crop).squeeze(0)
            laplacian_mask = laplacian_box[~mask[top_left.y : bottom_right.y + 1, top_left.x : bottom_right.x + 1]]
            return laplacian_mask.std() / 255

    def calculate_batched(
        self,
        deps: MetricBatchDependencies,
        image: ImageBatchTensor,
        annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        # max-laplacian = [4x255
        grayscale = deps["ephemeral_grayscale_image"]
        laplacian = laplacian2d(grayscale)
        if annotation is None:
            return torch.std(laplacian, dim=(1, 2, 3)) / 255.0
        else:
            torch.masked_select()
