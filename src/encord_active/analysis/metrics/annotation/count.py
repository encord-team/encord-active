from typing import Dict

from encord_active.analysis.metric import ImageObjectsMetric, MetricDependencies
from encord_active.analysis.types import AnnotationMetadata, ImageTensor, MetricResult
from encord_active.db.metrics import MetricType


class ObjectCountMetric(ImageObjectsMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_object_count",
            long_name="Object Count",
            desc="Number of objects present in an image",
            metric_type=MetricType.UINT,
        )

    def calculate(
        self,
        image: ImageTensor,
        image_deps: MetricDependencies,
        # key is object_hash | classification_hash
        annotations: Dict[str, AnnotationMetadata],
        annotations_deps: Dict[str, MetricDependencies],
    ) -> MetricResult:
        return len(annotations)
