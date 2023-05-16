from encord_active.analysis.context import ImageEvalContext
from encord_active.analysis.image import ImageContext
from encord_active.analysis.metric import LabelMetric
from encord_active.lib.common.utils import get_object_coordinates


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
        coordinates = get_object_coordinates(obj)
        x_coordinates = [x for x, _ in coordinates]
        y_coordinates = [y for _, y in coordinates]
        min_x = min(x_coordinates)
        max_x = max(x_coordinates)
        min_y = min(y_coordinates)
        max_y = max(y_coordinates)

        score = max(1 - min_x, 1 - min_y, max_x, max_y)
        return score
