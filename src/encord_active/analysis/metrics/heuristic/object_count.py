from typing import Dict

from encord_active.analysis.metric import ImageObjectsMetric, MetricDependencies, ObjectMetadata


class ObjectCountMetric(ImageObjectsMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='object-count',
            dependencies=set(),
            long_name='Image Aspect Ratio',
            short_desc='Number of objects present in an image',
            long_desc='Number of objects present in an image',
        )

    def calculate(self, img_deps: MetricDependencies, obj_deps: Dict[str, MetricDependencies],
                  objs: Dict[str, ObjectMetadata]) -> float:
        return len(objs)
