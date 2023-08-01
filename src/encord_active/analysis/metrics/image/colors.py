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
from encord_active.db.metrics import MetricType


class HSVColorMetric(OneImageMetric):
    def __init__(self, color_name: str, hue_query: float) -> None:
        """
        Computes the average hue distance to hue_query across all pixels in the
        image. The Hue query should be a value in range [0; 1] where 0 is red,
        1/3 is green and 2/3 is blue.

        Args:
            color_name:
            hue_query:
        """
        super().__init__(
            ident=f"metric_{color_name.lower()}",
            long_name=f"{color_name} Values".title(),
            desc=f"Ranks images by how {color_name.lower()} the average value of the image is.",
            metric_type=MetricType.NORMAL,
        )
        self.hue_query = hue_query * 2 * torch.pi

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        hsv_image = deps["ephemeral_hsv_image"]
        if not isinstance(hsv_image, torch.Tensor):
            raise ValueError("missing hsv image")

        if mask is None:
            hsv_pixels = hsv_image.reshape(3, -1)
        else:
            hsv_pixels = hsv_image[:, mask]

        hue_values = hsv_image[0]
        dists1 = (hue_values - self.hue_query) % 1.0
        dists2 = (self.hue_query - hue_values) % 1.0
        return torch.minimum(dists1, dists2).mean() * 2

    def calculate_batched(
        self, deps: MetricBatchDependencies, image: ImageBatchTensor, annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        hsv_image = deps["ephemeral_hsv_image"]
        hue_image = hsv_image[:, 0, :, :]

        # Convert to 'color'-ness normalised value
        hue_delta_raw = torch.abs(hue_image - self.hue_query)
        hue_delta_min = torch.min(hue_delta_raw, (2 * torch.pi) - hue_delta_raw)  # 0 -> torch.pi
        hue_delta = (torch.pi - hue_delta_min) / torch.pi  # 0 -> 1 (1 is 'color'-ness)

        objects = None
        if annotation is not None:
            masked_count = annotation.objects_deps["metric_area"].type(dtype=torch.float32)
            annotation_delta = torch.index_select(hue_delta, 0, annotation.objects_image_indices)
            masked_delta = torch.masked_fill(annotation_delta, ~annotation.objects_masks, 0)
            objects = torch.sum(masked_delta, dim=(1, 2)) / masked_count

        return ImageObjectOnlyOutputBatch(
            images=torch.mean(hue_delta, dim=(1, 2)),
            objects=objects,
        )
