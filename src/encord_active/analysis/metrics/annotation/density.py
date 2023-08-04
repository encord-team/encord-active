from typing import Dict

import torch

from encord_active.analysis.metric import ImageObjectsMetric, MetricDependencies
from encord_active.analysis.types import AnnotationMetadata, ImageTensor, MetricResult
from encord_active.analysis.util import image_height, image_width
from encord_active.db.metrics import MetricType


class ObjectDensityMetric(ImageObjectsMetric):
    def __init__(self) -> None:
        # FIXME: special case frame-level-annotation-metrics to a 3rd type of metric
        super().__init__(
            ident="metric_object_density",
            long_name="Object Density",
            desc="Number of objects present in an image",
            metric_type=MetricType.NORMAL,
        )

    def calculate(
        self,
        image: ImageTensor,
        image_deps: MetricDependencies,
        # key is object_hash | classification_hash
        annotations: Dict[str, AnnotationMetadata],
        annotations_deps: Dict[str, MetricDependencies],
    ) -> MetricResult:
        filter_annotations = [a.mask for a in annotations.values() if a.mask is not None]
        if len(filter_annotations) == 0:
            return 0.0
        mask_stack = torch.stack(filter_annotations)
        mask_union = torch.any(mask_stack, dim=0)
        mask_area = torch.sum(mask_union.type(torch.int64), dtype=torch.float32)
        img_area = float(image_width(image)) * float(image_height(image))
        # Relative area of mask union for non classification
        return mask_area / img_area
