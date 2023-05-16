from encord_active.analysis.context import ImageEvalContext
from encord_active.analysis.image import ImageContext
from encord_active.analysis.metric import LabelMetric


class AnnotationDuplicates(LabelMetric):
    def calculate_object(self, context: ImageEvalContext, image: ImageContext, obj: dict) -> float:
        ...