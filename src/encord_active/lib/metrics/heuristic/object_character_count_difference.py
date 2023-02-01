from collections import defaultdict

import easyocr
from cv2 import imread
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


class ObjectCharacterCountDifference(Metric):
    TITLE = "Object Character Count Difference"
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = [AnnotationType.OBJECT.BOUNDING_BOX]
    SHORT_DESCRIPTION = (
        "Counts the absolute difference between the number of text characters in the region covered by the object and "
        "the remaining objects from the same task."
    )
    LONG_DESCRIPTION = (
        "Counts the absolute difference between the number of text characters in the region covered by the object and "
        "the average number of text characters across all objects belonging to the same task."
    )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.ANNOTATION_TYPE}
        found_any = False
        reader = easyocr.Reader(["en"], verbose=False)

        du_hash_and_obj_hash_to_char_count = dict()
        label_hash_to_char_count_list = defaultdict(list)
        for data_unit, img_pth in iterator.iterate(desc="Counting text characters"):
            if img_pth is None:
                continue
            try:
                image = imread(str(img_pth))
                if image is None:
                    continue
            except Exception:
                continue

            H, W = image.shape[:2]
            for obj in data_unit["labels"].get("objects", []):
                if not obj["shape"] in valid_annotation_types:
                    continue
                coords = get_object_coordinates(obj)
                if coords is None:
                    continue
                x1, y1 = int(coords[0][0] * W), int(coords[0][1] * H)
                x2, y2 = int(coords[2][0] * W), int(coords[2][1] * H)
                try:
                    assert 0 <= x1 <= W
                    assert 0 <= y1 <= H
                    assert 0 <= x2 <= W
                    assert 0 <= y2 <= H
                except Exception as e:
                    logger.info(e)
                    continue

                bbox = (x1, y1, x2, y2)

                try:
                    text = reader.readtext(self._imcrop(image, bbox), detail=0)
                except Exception as e:
                    logger.info(
                        f"Exception found in label row {iterator.label_hash}, data unit {iterator.du_hash} and object hash {obj['objectHash']}"
                    )
                    continue
                char_cnt = sum(map(len, text))
                du_hash_and_obj_hash_to_char_count[(iterator.du_hash, obj["objectHash"])] = char_cnt
                label_hash_to_char_count_list[iterator.label_hash].append(char_cnt)
                found_any = True

        # Calculate average number of text characters across all objects belonging to the same task (label_hash)
        label_hash_to_avg_char_count = {
            label_hash: sum(char_count_list) / len(char_count_list)
            for label_hash, char_count_list in label_hash_to_char_count_list.items()
        }

        for data_unit, _ in iterator.iterate(desc="Counting text characters"):
            for obj in data_unit["labels"].get("objects", []):
                compound_hash = (iterator.du_hash, obj["objectHash"])
                if compound_hash in du_hash_and_obj_hash_to_char_count:
                    abs_diff = abs(
                        du_hash_and_obj_hash_to_char_count[compound_hash]
                        - label_hash_to_avg_char_count[iterator.label_hash]
                    )
                    writer.write(abs_diff, obj)

        if not found_any:
            logger.info(
                f"<yellow>[Skipping]</yellow> No object labels of types {{{', '.join(valid_annotation_types)}}}."
            )

    @staticmethod
    def _imcrop(image, bbox):
        x1, y1, x2, y2 = bbox
        H, W = image.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, W), min(y2, H)
        return image[y1:y2, x1:x2, :]
