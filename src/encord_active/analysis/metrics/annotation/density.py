from typing import Dict
import torch
from encord_active.analysis.metric import (
    ImageObjectsMetric,
    MetricDependencies,
)
from encord_active.analysis.types import MetricResult, AnnotationMetadata, ImageTensor
from encord_active.analysis.util import image_width, image_height


class ObjectDensityMetric(ImageObjectsMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_object_density",
            dependencies=set(),
            long_name="Object Density",
            desc="Number of objects present in an image",
        )

    def calculate(
        self,
        image: ImageTensor,
        image_deps: MetricDependencies,
        # key is object_hash | classification_hash
        annotations: Dict[str, AnnotationMetadata],
        annotations_deps: Dict[str, MetricDependencies],
    ) -> MetricResult:
        if len(annotations) == 0:
            return 0.0
        mask_stack = torch.stack([a.mask for a in annotations.values()])
        mask_union = torch.any(mask_stack, dim=0)
        mask_area = torch.sum(mask_union.type(torch.int64), dtype=torch.float32)
        img_area = float(image_width(image)) * float(image_height(image))
        # Relative area of mask union for non classification
        return mask_area / img_area
