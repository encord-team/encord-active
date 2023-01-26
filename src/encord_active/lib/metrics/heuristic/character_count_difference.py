from collections import defaultdict

import easyocr

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


class CharacterCountDifference(Metric):
    TITLE = "Character Count Difference"
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.NONE
    SHORT_DESCRIPTION = (
        "Counts the absolute difference between the number of text characters in the image and the remaining images "
        "from the same task."
    )
    LONG_DESCRIPTION = (
        "Counts the absolute difference between the number of text characters in the image and the average number of "
        "text characters across all the images belonging to the same task."
    )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        reader = easyocr.Reader(["en"], verbose=False)

        du_hash_to_char_count = dict()
        label_hash_to_char_count_list = defaultdict(list)
        for _, img_pth in iterator.iterate(desc="Counting text characters"):
            if img_pth is None:
                continue
            text = reader.readtext(img_pth.as_posix(), detail=0)
            char_cnt = sum(map(len, text))

            du_hash_to_char_count[iterator.du_hash] = char_cnt
            label_hash_to_char_count_list[iterator.label_hash].append(char_cnt)

        # Calculate average number of text characters across all images belonging to the same task (label_hash)
        label_hash_to_avg_char_count = {
            label_hash: sum(char_count_list) / len(char_count_list)
            for label_hash, char_count_list in label_hash_to_char_count_list.items()
        }

        for _, img_pth in iterator.iterate(desc="Writing scores"):
            if img_pth is None:
                continue
            abs_diff = abs(du_hash_to_char_count[iterator.du_hash] - label_hash_to_avg_char_count[iterator.label_hash])
            writer.write(abs_diff)
