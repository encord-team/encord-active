from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


class ObjectsCountMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Object Count",
            short_description="Counts number of objects in the image",
            long_description=r"""Counts number of objects in the image.""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.ALL,
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        if not iterator.project.ontology.objects:
            return

        for data_unit, img_pth in iterator.iterate(desc="Counting objects"):
            score = len(data_unit["labels"]["objects"]) if "objects" in data_unit["labels"] else 0
            writer.write(score)
