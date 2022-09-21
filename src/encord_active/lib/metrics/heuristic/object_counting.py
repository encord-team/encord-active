from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.metric import AnnotationType, DataType, Metric, MetricType
from encord_active.lib.common.writer import CSVMetricWriter


class ObjectsCountMetric(Metric):
    TITLE = "Object Count"
    SHORT_DESCRIPTION = "Counts number of objects in the image"
    LONG_DESCRIPTION = r"""Counts number of objects in the image."""
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.ALL

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        for data_unit, img_pth in iterator.iterate(desc="Counting objects"):
            score = len(data_unit["labels"]["objects"]) if "objects" in data_unit["labels"] else 0
            writer.write(score)
