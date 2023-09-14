from typing import Dict, Optional, cast

import torch
from shapely.geometry import Polygon

from encord_active.analysis.metric import (
    MetricDependencies,
    TemporalObjectByFrameMetric,
)
from encord_active.analysis.types import AnnotationMetadata, MetricResult
from encord_active.db.metrics import MetricType
from encord_active.lib.common.utils import get_polygon, mask_iou


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
            ident="metric_missing_or_broken_track",
            long_name="Missing Object, Wrong Track",
            desc="",
            metric_type=MetricType.NORMAL,
        )

    def calculate(
        self,
        annotations: Dict[str, AnnotationMetadata],
        annotation_deps: dict[str, MetricDependencies],
        prev_annotations: Optional[Dict[str, AnnotationMetadata]],
        next_annotations: Optional[Dict[str, AnnotationMetadata]],
    ) -> Dict[str, MetricResult]:
        annotation_res: Dict[str, MetricResult] = {}
        if not prev_annotations or not next_annotations:
            return {annotation_hash: 1.0 for annotation_hash in annotations.keys()}

        for next_annotation_hash, next_annotation_meta in next_annotations.items():
            iou = None

            if next_annotation_hash in prev_annotations.keys():
                if prev_annotations[next_annotation_hash].mask is None or next_annotation_meta.mask is None:
                    iou = 0.0
                else:
                    iou = mask_iou(prev_annotations[next_annotation_hash].mask, next_annotation_meta.mask)

            # Check if prev and next annotation is temporally connected
            if iou is not None and iou > 0.5:
                # Check if there is a missing annotation in the current frame
                if next_annotation_hash not in annotations.keys():
                    annotation_res[next_annotation_hash] = 0.0
                else:
                    current_annotation = annotations[next_annotation_hash]
                    prev_iou = mask_iou(current_annotation.mask, prev_annotations[next_annotation_hash].mask)
                    next_iou = mask_iou(current_annotation.mask, next_annotation_meta.mask)
                    if prev_iou < 0.5 or next_iou < 0.5:
                        annotation_res[next_annotation_hash] = min(prev_iou, next_iou)
            else:
                annotation_res[next_annotation_hash] = 1.0

        return annotation_res

        # for prev_annotation_hash, prev_annotation_meta in prev_annotations.items():
        #     if prev_annotation_meta.feature_hash == next_annotation_meta.feature_hash:
        #         iou = None
        #         if prev_annotation_meta.mask is None and next_annotation_meta.mask is None:
        #             iou = 1.0
        #         elif prev_annotation_meta.mask is not None and next_annotation_meta.mask is not None:
        #             iou = mask_iou(prev_annotation_meta.mask, next_annotation_meta.mask)
        #         if iou is not None and (best_iou is None or iou > best_iou):
        #             best_iou = iou

        # If iou > 0.5 we consider this to be a object that is likely to be interpolated in the middle
        #     if iou is not None and iou > 0.5:
        #         # Search for best iou in the middle
        #         best_mid_iou = 0.0
        #
        #         # Check if there is a missing annotation in this track
        #         if next_annotation_hash not in annotations.keys():
        #             annotation_res[next_annotation_hash] = 0.0
        #         else:
        #             current_annotation = annotations[next_annotation_hash]
        #             pass
        #
        #         best_mid_annotation_hash = list(annotations.keys())[0]  # FIXME: may error!?
        #         for annotation_hash, annotation_meta in annotations.items():
        #             if annotation_meta.feature_hash == next_annotation_meta.feature_hash:
        #                 pass  # FIXME: implement
        #
        #         # FIXME: kinda hard to select best annotation hash to union with if multiple map to this hash
        #         # FIXME: works even worse if 0 items exist on this frame!?!
        #         annotation_res[best_mid_annotation_hash] = min(
        #             best_mid_iou, cast(float, annotation_res.get(best_mid_annotation_hash, 1.0))
        #         )
        #         # image_difficulty (KMeans)
        #         # shape outlier = _hu_static (Ignore??)
        #         #
        #
        # # Any annotations not marked as suspect get instant scores of 1.0
        # for annotation_hash in annotations.keys():
        #     if annotation_hash not in annotation_res:
        #         annotation_res[annotation_hash] = 1.0
        # return annotation_res
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
