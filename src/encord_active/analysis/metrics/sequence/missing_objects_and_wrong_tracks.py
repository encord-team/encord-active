from typing import Dict, List, Tuple, Optional

from shapely.geometry import Polygon

from encord_active.analysis.metric import (
    MetricDependencies,
    TemporalObjectByFrameMetric,
)
from encord_active.analysis.types import AnnotationMetadata
from encord_active.lib.common.utils import get_iou, get_polygon


def _filter_object(obj: Dict[str, MetricDependencies]) -> Dict[str, Polygon]:
    return {
        k: get_polygon(v) for k, v in obj.items() if v["shape"] in {"bounding_box", "rotatable_bounding_box", "polygon"}
    }


class TemporalMissingObjectsAndWrongTracks(TemporalObjectByFrameMetric):
    """
    Temporal analysis of missing objects between frame dependencies
    """

    def __init__(self) -> None:
        super().__init__(
            ident="metric_label_missing_or_broken_tracks",
            dependencies=set(),
            long_name="Missing Object, Wrong Track",
            desc="",
        )

    def calculate(
        self,
        annotations: Dict[str, AnnotationMetadata],
        annotation_deps: dict[str, MetricDependencies],
        prev_annotations: Optional[Dict[str, AnnotationMetadata]],
        next_annotations: Optional[Dict[str, AnnotationMetadata]],
    ) -> Dict[str, float]:
        return {}
        """
        if len(prev_frames) == 0 or len(next_frames) == 0:
            return {k: 0.0 for k in objs.keys()}
        prev_objects = _filter_object(prev_frames[0][1])
        curr_objects = _filter_object(objs)
        next_objects = _filter_object(next_frames[0][1])
        check_objects = set(prev_objects.keys()) & set(next_objects.keys())
        iou_objects = {}
        best_iou = 0
        for check_obj in check_objects:
            prev_poly = prev_objects[check_obj]
            next_poly = next_objects[check_obj]
            iou = get_iou(prev_poly, next_poly)
            iou_objects[check_objects] = iou
            best_iou = max(best_iou, iou)
        """

        # FIXME: finish implementation of this
        #  and potentially define variant of logic that requires that objectHash remains consistent?
        #  optional warn on looks like same object temporal but is actually different.
