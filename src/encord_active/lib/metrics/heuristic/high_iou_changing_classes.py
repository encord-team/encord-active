from loguru import logger

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import get_iou, get_polygon
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


class HighIOUChangingClasses(Metric):
    def __init__(self, threshold: float = 0.8):
        super(HighIOUChangingClasses, self).__init__(
            title="Inconsistent Object Classification and Track IDs",
            short_description="Looks for overlapping objects with different classes (across frames).",
            long_description=r"""This algorithm looks for overlapping objects in consecutive
frames that have different classes. Furthermore, if classes are the same for overlapping objects but have different
track-ids, they will be flagged as potential inconsistencies in tracks.


**Example 1:**
```
      Frame 1                       Frame 2
┌───────────────────┐        ┌───────────────────┐
│                   │        │                   │
│  ┌───────┐        │        │  ┌───────┐        │
│  │       │        │        │  │       │        │
│  │ CAT:1 │        │        │  │ DOG:1 │        │
│  │       │        │        │  │       │        │
│  └───────┘        │        │  └───────┘        │
│                   │        │                   │
└───────────────────┘        └───────────────────┘
```
`Dog:1` will be flagged as potentially wrong class, because it overlaps with `CAT:1`.

**Example 2:**
```
      Frame 1                       Frame 2
┌───────────────────┐        ┌───────────────────┐
│                   │        │                   │
│  ┌───────┐        │        │  ┌───────┐        │
│  │       │        │        │  │       │        │
│  │ CAT:1 │        │        │  │ CAT:2 │        │
│  │       │        │        │  │       │        │
│  └───────┘        │        │  └───────┘        │
│                   │        │                   │
└───────────────────┘        └───────────────────┘
```
`Cat:2` will be flagged as potentially having a broken track, because track ids `1` and `2` doesn't match.

""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.SEQUENCE,
            annotation_type=[AnnotationType.OBJECT.BOUNDING_BOX, AnnotationType.OBJECT.POLYGON],
        )
        self.threshold = threshold

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
        found_any = False
        found_valid = False

        label_hash = ""
        previous_objects = None
        previous_polygons = None
        for data_unit, img_pth in iterator.iterate(desc="Looking for overlapping objects"):
            label_row = iterator.label_rows[iterator.label_hash]
            data_type = label_row["data_type"]
            if not (data_type == "video" or (data_type == "img_group" and len(label_row["data_units"]) > 1)):
                # Not a sequence
                continue

            objects = [o for o in data_unit["labels"]["objects"] if o["shape"] in valid_annotation_types]
            polygons = list(map(get_polygon, objects))
            found_any |= len(polygons) > 0

            # Remove invalid polygons and flag them as such.
            for i in range(len(objects) - 1, -1, -1):
                if polygons[i] is None:
                    writer.write(0.0, objects[i], description="Invalid polygon")
                    objects.pop(i)
                    polygons.pop(i)

            if label_hash != iterator.label_hash or previous_objects is None:
                label_hash = iterator.label_hash
                previous_objects = objects
                previous_polygons = polygons
                continue

            for obj, polygon in zip(objects, polygons):
                best_idx = -1
                best_iou = 0.0

                for i, old_polygon in enumerate(previous_polygons):
                    iou = get_iou(polygon, old_polygon)
                    if iou > best_iou:
                        best_idx = i
                        best_iou = iou

                if best_iou == 0 or best_idx == -1:
                    writer.write(1.0, obj)
                    continue

                prev_object = previous_objects[best_idx]
                if prev_object["objectHash"] == obj["objectHash"]:
                    writer.write(1.0, obj)
                elif best_iou > self.threshold and prev_object["featureHash"] != obj["featureHash"]:
                    # Overlapping objects with different classes
                    writer.write(
                        1 - best_iou,
                        obj,
                        description=f"`{obj['name']}` in frame {iterator.frame} overlaps with "
                        f"`{prev_object['name']}` in frame {iterator.frame - 1}",
                    )
                elif best_iou > self.threshold and prev_object["featureHash"] == obj["featureHash"]:
                    writer.write(
                        1 - best_iou,
                        obj,
                        description=f"`{obj['name']}` in frame {iterator.frame - 1} and {iterator.frame} have "
                        f"different track ids.",
                    )
                else:
                    # IOU < threshold so probably not an issue
                    writer.write(1.0, obj)
                found_valid = True

            previous_polygons = polygons
            previous_objects = objects

        if not found_any:
            logger.info(
                f"<yellow>[Skipping]</yellow> No object labels of types {{{', '.join(valid_annotation_types)}}}."
            )
        elif not found_valid:
            logger.info(
                f"<yellow>[Skipping]</yellow> No valid object labels of types {{{', '.join(valid_annotation_types)}}}."
            )
