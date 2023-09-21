from typing import Dict, Optional

from shapely.geometry import Polygon

from encord_active.analysis.metric import (
    MetricDependencies,
    MetricResultOptionalAnnotation,
    TemporalObjectByFrameMetric,
)
from encord_active.analysis.types import AnnotationMetadata
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
            long_name="Wrong Object in Track",
            desc="",
            metric_type=MetricType.NORMAL,
        )

    def calculate(
        self,
        annotations: Dict[str, AnnotationMetadata],
        annotation_deps: dict[str, MetricDependencies],
        prev_annotations: Optional[Dict[str, AnnotationMetadata]],
        next_annotations: Optional[Dict[str, AnnotationMetadata]],
    ) -> Dict[str, MetricResultOptionalAnnotation]:
        annotation_res: Dict[str, MetricResultOptionalAnnotation] = {}

        prev_annotations = {} if prev_annotations is None else prev_annotations
        next_annotations = {} if next_annotations is None else next_annotations

        if not annotations:
            return annotation_res

        if (not prev_annotations) or (not next_annotations):
            return {annotation_hash: 1.0 for annotation_hash in annotations.keys()}

        for next_annotation_hash, next_annotation_meta in next_annotations.items():
            if next_annotation_meta.mask is None:
                continue

            iou = None
            prev_annotation_mask = None

            if next_annotation_hash in prev_annotations.keys():
                prev_annotation_mask = prev_annotations[next_annotation_hash].mask
                if prev_annotation_mask is None:
                    continue
                else:
                    iou = mask_iou(prev_annotation_mask, next_annotation_meta.mask)

            # Check if prev and next annotation is temporally connected
            if iou is not None and iou > 0.5:
                for current_annotation_hash, current_annotation_meta in annotations.items():
                    if current_annotation_meta.mask is None or prev_annotation_mask is None:
                        annotation_res[current_annotation_hash] = 0.0
                    else:
                        prev_iou = mask_iou(current_annotation_meta.mask, prev_annotation_mask)
                        next_iou = mask_iou(current_annotation_meta.mask, next_annotation_meta.mask)

                        # wrong object
                        if prev_iou > 0.5 and next_iou > 0.5:
                            if current_annotation_hash == next_annotation_hash:
                                annotation_res[current_annotation_hash] = 1.0
                            else:
                                annotation_res[current_annotation_hash] = 0.0

        for annotation_hash in annotations.keys():
            if annotation_hash not in annotation_res:
                annotation_res[annotation_hash] = 1.0

        return annotation_res
