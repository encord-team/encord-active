import numpy as np

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


class RandomeImageMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Random Values on Images",
            short_description="Assigns a random value between 0 and 1 to images",
            long_description="Uses a uniform distribution to generate a value between 0 and 1 to each image",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=[],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}

        for _ in iterator.iterate(desc="Assigning random values to images"):
            writer.write(np.random.uniform())


class RandomeObjectMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Random Values on Objects",
            short_description="Assigns a random value between 0 and 1 to objects",
            long_description="Uses a uniform distribution to generate a value between 0 and 1 to each object",
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

        for data_unit, _ in iterator.iterate(desc="Assigning random values to objects"):
            for obj in data_unit["labels"].get("objects", []):
                if not obj["shape"] in valid_annotation_types:
                    continue
                writer.write(np.random.uniform(), obj)
