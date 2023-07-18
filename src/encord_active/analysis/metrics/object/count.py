from typing import Dict

from encord_active.analysis.metric import (
    ImageObjectsMetric,
    MetricDependencies,
)
from encord_active.analysis.types import MetricResult, AnnotationMetadata, ImageTensor


class ObjectCountMetric(ImageObjectsMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="object-count",
            dependencies=set(),
            long_name="Object Count",
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
        return len(annotations)
