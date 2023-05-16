from encord_active.analysis.context import ImageEvalContext
from encord_active.analysis.image import ImageContext
from encord_active.analysis.metric import ImageMetric


class ObjectCountMetric(ImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='object-count',
            dependencies=set(),
            long_name='Image Aspect Ratio',
            short_desc='',
            long_desc='',
        )

    def calculate(self, context: ImageEvalContext, image: ImageContext) -> float:
        return float(len(image.objects()))
