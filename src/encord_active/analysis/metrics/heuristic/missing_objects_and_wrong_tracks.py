from typing import Tuple, List, Dict

from shapely.geometry import Polygon

from encord_active.analysis.metric import TemporalObjectByFrameMetric, MetricDependencies
from encord_active.analysis.types import ObjectMetadata
from encord_active.lib.common.utils import get_iou, get_polygon

def _filter_object(obj: Dict[str, MetricDependencies]) -> Dict[str, Polygon]:
    return {
        k: get_polygon(v)
        for k, v in obj.items()
        if v["shape"] in {"bounding_box", "rotatable_bounding_box", "polygon"}
    }


class MissingObjectsAndWrongTracks(TemporalObjectByFrameMetric):
    """
    Temporal analysis of missing objects between frame dependencies
    """
    def __init__(self) -> None:
        super().__init__(
            ident='missing-object-wrong-track',
            dependencies=set(),
            long_name="Missing Object, Wrong Track",
            short_desc="",
            long_desc="",
            prev_frame_count=1,
            next_frame_count=1)

    def calculate(self, img_deps: MetricDependencies, obj_deps: Dict[str, MetricDependencies],
                  objs: Dict[str, ObjectMetadata],
                  prev_frames: List[
                      Tuple[MetricDependencies, Dict[str, MetricDependencies], Dict[str, ObjectMetadata]]
                  ],
                  next_frames: List[
                      Tuple[MetricDependencies, Dict[str, MetricDependencies], Dict[str, ObjectMetadata]]
                  ],
                  ) -> Dict[str, float]:
        if len(prev_frames) == 0 or len(next_frames) == 0:
            return {
                k: 0.0
                for k in objs.keys()
            }
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

        # FIXME: finish implementation of this
        #  and potentially define variant of logic that requires that objectHash remains consistent?
        #  optional warn on looks like same object temporal but is actually different.