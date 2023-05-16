from encord_active.analysis.context import ImageEvalContext
from encord_active.analysis.image import ImageContext
from encord_active.analysis.metric import LabelMetric
from encord_active.lib.metrics.geometric.object_size import get_area


class ImageBorderCloseness(LabelMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='image-border-closeness',
            dependencies=set(),
            long_name='Image Border Closeness',
            short_desc='',
            long_desc='',
        )

    def calculate_object(self, context: ImageEvalContext, image: ImageContext, obj: dict) -> float:
        obj_area = get_area(obj)
        return obj_area * 100.0
