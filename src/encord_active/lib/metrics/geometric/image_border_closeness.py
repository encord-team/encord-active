from loguru import logger

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import get_object_coordinates
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


class ImageBorderCloseness(Metric):
    def __init__(self):
        super().__init__(
            title="Annotation closeness to image borders",
            short_description="Ranks annotations by how close they are to image borders.",
            long_description=r"""This metric ranks annotations by how close they are to image borders.""",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
                AnnotationType.OBJECT.POLYLINE,
                AnnotationType.OBJECT.KEY_POINT,
                AnnotationType.OBJECT.SKELETON,
            ],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
        found_any = False

        for data_unit, _ in iterator.iterate(desc="Computing closeness to border"):
            for obj in data_unit["labels"].get("objects", []):
                if obj["shape"] not in valid_annotation_types:
                    continue

                coordinates = get_object_coordinates(obj)
                if not coordinates:  # avoid corrupted objects without vertices ([]) and unknown objects' shape (None)
                    continue

                x_coordinates = [x for x, _ in coordinates]
                min_x = min(x_coordinates)
                max_x = max(x_coordinates)

                y_coordinates = [y for _, y in coordinates]
                min_y = min(y_coordinates)
                max_y = max(y_coordinates)

                score = max(1 - min_x, 1 - min_y, max_x, max_y)
                writer.write(score, obj)
                found_any = True

        if not found_any:
            logger.info(
                f"<yellow>[Skipping]</yellow> No object labels of types {{{', '.join(valid_annotation_types)}}}."
            )
