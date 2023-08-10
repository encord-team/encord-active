from typing import Optional

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
from encord_active.db.metrics import MetricType


class ContrastMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_contrast",
            long_name="Contrast",
            desc="",
            metric_type=MetricType.NORMAL,
        )

    def calculate(
        self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor], bb: Optional[BoundingBoxTensor]
    ) -> MetricResult:
        # Max-std : [0, 255] = 255/2 = 127.5
        image_reduced = torch.round(torch.mean(image, dim=0, dtype=torch.float32))
        if mask is None or bb is None:
            return torch.std(image_reduced) / 127.5
        else:
            # Masked standard deviation - using known correct impl to fix later.
            # mask_mean = float(deps["metric_brightness"])
            # mask_count = float(deps["metric_area"])

            # mask_mean_delta = torch.masked_fill(image_reduced - mask_mean, ~mask, 0)
            # mask_mean_delta_sq_sum = torch.sum(mask_mean_delta**2, dtype=torch.float32)

            # return torch.sqrt(mask_mean_delta_sq_sum / mask_count) / 127.5
            x1, y1, x2, y2 = bb.type(torch.int32).tolist()
            image_reduced = image_reduced[y1 : y2 + 1, x1 : x2 + 1]
            mask = mask[y1 : y2 + 1, x1 : x2 + 1]
            return torch.masked.std(image_reduced, mask=mask) / 127.5

    def calculate_batched(
        self, deps: MetricBatchDependencies, image: ImageBatchTensor, annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        grayscale: ImageBatchTensor = deps["ephemeral_grayscale_image"]

        objects = None
        if annotation is not None:
            masked_mean = annotation.objects_deps["metric_brightness"].type(dtype=torch.float32) * 255.0
            masked_count = annotation.objects_deps["metric_area"].type(dtype=torch.float32)
            mask_grayscale = torch.index_select(image, 0, annotation.objects_image_indices)
            masked_mean_resize = masked_mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            mask_mean_delta = mask_grayscale.type(dtype=torch.float32) - masked_mean_resize
            masked_delta = torch.masked_fill(mask_mean_delta, ~annotation.objects_masks, 0)
            masked_sum_sq = torch.sum(torch.square(masked_delta), dtype=torch.float32).type(dtype=torch.float32)
            objects = (masked_sum_sq / masked_count) / 127.5

        return ImageObjectOnlyOutputBatch(
            images=torch.std(grayscale.type(dtype=torch.float32), dim=(-1, -2)) / 127.5,
            objects=objects,
        )
