import numpy as np
from loguru import logger

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import Metric
from encord_active.lib.metrics.types import AnnotationType, DataType, MetricType
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


class RandomImageMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Random Values on Images",
            short_description="Assigns a random value between 0 and 1 to images",
            long_description="Uses a uniform distribution to generate a value between 0 and 1 to each image",
            doc_url="https://docs.encord.com/docs/active-data-quality-metrics#random-values-on-images",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        for _ in iterator.iterate(desc="Assigning random values to images"):
            writer.write(np.random.uniform())


class RandomObjectMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Random Values on Objects",
            short_description="Assigns a random value between 0 and 1 to objects",
            long_description="Uses a uniform distribution to generate a value between 0 and 1 to each object",
            doc_url="https://docs.encord.com/docs/active-label-quality-metrics#random-values-on-objects",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
            ],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}

        found_any = False
        for data_unit, _ in iterator.iterate(desc="Searching for objects and assigning random scores"):
            for obj in data_unit["labels"].get("objects", []):
                if not obj["shape"] in valid_annotation_types:
                    continue
                found_any = True
                writer.write(np.random.uniform(), obj)

        if not found_any:
            logger.info(
                f"<yellow>[Skipping]</yellow> No object labels of types {{{', '.join(valid_annotation_types)}}}."
            )
