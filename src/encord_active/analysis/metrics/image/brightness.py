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
from encord_active.db.metrics import MetricType


class BrightnessMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_brightness",
            long_name="Brightness",
            desc="",
            metric_type=MetricType.NORMAL,
        )

    def calculate(
        self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor], bb: Optional[BoundingBoxTensor]
    ) -> MetricResult:
        if mask is None or bb is None:
            return torch.mean(image, dtype=torch.float) / 255.0
        else:
            x1, y1, x2, y2 = bb.type(torch.int32).tolist()
            image = image[:, y1 : y2 + 1, x1 : x2 + 1]
            mask = mask[y1 : y2 + 1, x1 : x2 + 1]
            mask_count = float(cast(float, deps["metric_area"]))
            image_reduced = torch.round(torch.mean(image, dim=0, dtype=torch.float32))
            mask_total = torch.sum(torch.masked_fill(image_reduced, ~mask, 0), dtype=torch.float32).item()
            return float(mask_total) / (255.0 * mask_count)

    def calculate_batched(
        self, deps: MetricBatchDependencies, image: ImageBatchTensor, annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        grayscale = deps["ephemeral_grayscale_image"]
        objects = None
        if annotation is not None:
            masked_count = annotation.objects_deps["metric_area"].type(dtype=torch.float32)
            mask_grayscale = torch.index_select(image, 0, annotation.objects_image_indices)
            masked_gray = torch.masked_fill(mask_grayscale, ~annotation.objects_masks, 0)
            masked_sum = torch.sum(masked_gray, dtype=torch.int64).type(dtype=torch.float32)
            objects = masked_sum / masked_count

        return ImageObjectOnlyOutputBatch(
            images=torch.mean(grayscale, dim=(-1, -2), dtype=torch.float32) / 255.0, objects=objects
        )
