import easyocr

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


class CharacterCount(Metric):
    TITLE = "Character Count"
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.NONE
    SHORT_DESCRIPTION = "Counts number of text characters in the image."
    LONG_DESCRIPTION = "Counts number of text characters in the image."

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        reader = easyocr.Reader(["en"], verbose=False)

        for data_unit, img_pth in iterator.iterate(desc="Counting text characters"):
            text = reader.readtext(str(img_pth), detail=0)
            char_cnt = sum(map(len, text))
            writer.write(char_cnt)
