"""
Detects when objects with sufficient iou are missing in "in-between" frames.
"""

from typing import List, Tuple, Union

from loguru import logger
from shapely.geometry import Polygon

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


class ErrorStore:
    """
    "Dict" that stores the lowest value if value exists already.
    """

    def __init__(self):
        self.errors = {}

    def add(self, object: dict, frame: int, score: Union[int, float], description=""):
        key = (object["objectHash"], frame)
        if key not in self.errors or self.errors[key][1] > score:
            self.errors[key] = (object, score, description)


class MissingObjectsMetric(Metric):
    def __init__(self, threshold: float = 0.5):
        super().__init__(
            title="Missing Objects and Broken Tracks",
            short_description="Identifies missing objects and broken tracks based on object overlaps.",
            long_description=r"""Identifies missing objects by comparing object overlaps based
on a running window.

**Case 1:**
If an intermediate frame (frame $i$) doesn't include an object in the same
region, as the two surrounding framge ($i-1$ and $i+1$), it is flagged.

```
      Frame i-1                     Frame i                      Frame i+1
┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
│                   │        │                   │        │                   │
│  ┌───────┐        │        │                   │        │  ┌───────┐        │
│  │       │        │        │                   │        │  │       │        │
│  │ CAT:1 │        │        │                   │        │  │ CAT:1 │        │
│  │       │        │        │                   │        │  │       │        │
│  └───────┘        │        │                   │        │  └───────┘        │
│                   │        │                   │        │                   │
│                   │        │                   │        │                   │
└───────────────────┘        └───────────────────┘        └───────────────────┘
```
Frame $i$ will be flagged as potentially missing an object.

**Case 2:**
If objects of the same class overlap in three consecutive frames ($i-1$, $i$, and $i+1$) but do not share object
hash, they will be flagged as a potentially broken track.

```
      Frame i-1                     Frame i                      Frame i+1
┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
│                   │        │                   │        │                   │
│  ┌───────┐        │        │  ┌───────┐        │        │  ┌───────┐        │
│  │       │        │        │  │       │        │        │  │       │        │
│  │ CAT:1 │        │        │  │ CAT:2 │        │        │  │ CAT:1 │        │
│  │       │        │        │  │       │        │        │  │       │        │
│  └───────┘        │        │  └───────┘        │        │  └───────┘        │
│                   │        │                   │        │                   │
│                   │        │                   │        │                   │
└───────────────────┘        └───────────────────┘        └───────────────────┘
```
`CAT:2` will be marked as potentially having a wrong track id.
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
        found_sequential = False
        error_store = ErrorStore()
        # Prepare sliding window of previous two frames to compare polygons over time
        window: List[List[Tuple[dict, Polygon]]] = []

        for data_unit, img_pth in iterator.iterate(desc="Looking for broken tracks"):
            label_row = iterator.label_rows[iterator.label_hash]
            frame = iterator.frame

            data_type = label_row["data_type"]
            if not (data_type == "video" or (data_type == "img_group" and len(label_row["data_units"]) > 1)):
                # Not a sequence
                continue
            found_sequential = True

            objects = [o for o in data_unit["labels"]["objects"] if o["shape"] in valid_annotation_types]
            polygons = list(map(get_polygon, objects))

            if len(polygons) == 0:
                continue
            found_any = True

            # Remove invalid polygons and flag them as such in the csv file.
            for i in range(len(objects) - 1, -1, -1):
                if polygons[i] is None:
                    error_store.add(object=objects[i], frame=frame, score=0.0)
                    objects.pop(i)
                    polygons.pop(i)

            if len(objects) == 0:
                continue
            found_valid = True

            # First two iterations just populates the window.
            if len(window) < 2:
                window.append(list(zip(objects, polygons)))
                continue

            # Idea:
            # 1. Find best IOU match from two frames back (of same class).
            # 2. If there is a sufficiently good match, check that there is also a good
            #    match in the in-between frame.
            # 3. If not, flag it as potentially missing.
            for obj, poly in zip(objects, polygons):
                best_iou = 0
                best_object: dict = {}

                # 1. Look for highest IOU two frames back
                for old_obj, old_poly in window[0]:
                    if old_obj["value"] != obj["value"]:
                        continue

                    iou = get_iou(poly, old_poly)
                    if iou > best_iou and old_obj["value"] == obj["value"]:
                        best_iou = iou
                        best_object = old_obj

                if best_iou < self.threshold / 2:  # Skip objects that are not similar enough.
                    error_store.add(object=obj, frame=frame, score=1.0)
                    continue

                # 2. Look for the in-between match
                best_mid_iou = 0
                best_mid_hash = ""
                for mid_obj, mid_poly in window[1]:
                    if mid_obj["featureHash"] != obj["featureHash"]:
                        continue

                    iou = get_iou(poly, mid_poly)
                    if iou > best_mid_iou:
                        best_mid_iou = iou
                        best_mid_hash = mid_obj["objectHash"]

                if best_mid_iou > self.threshold:
                    # We'll only execute this when best_object is actually an object due to `continue` in line 144
                    object_hashes = {
                        best_object.get("objectHash"),
                        best_mid_hash,
                        obj["objectHash"],
                    }
                    if len(object_hashes) > 1:
                        error_store.add(
                            object=obj,
                            frame=frame,
                            score=0.5,
                            description=f"Track may be broken between frames [{iterator.frame - 2}:{iterator.frame}]",
                        )
                    else:
                        error_store.add(object=obj, frame=frame, score=1.0)
                    continue

                # 3. If missing, flag.
                error_store.add(
                    object=best_object,
                    frame=frame - 2,
                    score=0.0,
                    description=f"`{obj['value']}` could be missing on next frame ({iterator.frame-1}).",
                )

            window.pop(0)
            window.append(list(zip(objects, polygons)))

        if not found_sequential:
            logger.info("<yellow>[Skipping]</yellow> No sequential data found.")
            return

        if not found_any:
            logger.info(
                f"<yellow>[Skipping]</yellow> No object labels of types {{{', '.join(valid_annotation_types)}}}."
            )
            return
        elif not found_valid:
            logger.info(
                f"<yellow>[Skipping]</yellow> No valid object labels of types {{{', '.join(valid_annotation_types)}}}."
            )
            return

        # Collect the results in the CSV file.
        # Everything not found above with get score "1" meaning "no issues".
        annotated = {k: False for k in error_store.errors}
        for data_unit, _ in iterator.iterate(desc="Storing results"):
            for obj in data_unit["labels"].get("objects", []):
                key = (obj["objectHash"], iterator.frame)
                if key in error_store.errors:
                    annotated[key] = True
                    _obj, score, desc = error_store.errors[key]
                    writer.write(score, _obj, description=desc)

        if not all(annotated.values()):
            logger.warning("Found errors that were not logged. This should not have happened.")
            for k, v in annotated.items():
                if not v:
                    logger.info(k)
