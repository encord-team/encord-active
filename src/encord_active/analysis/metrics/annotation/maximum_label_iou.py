from typing import Dict, Tuple

import torch

from encord_active.analysis.metric import (
    MetricDependencies,
    MetricResultOptionalAnnotation,
    ObjectByFrameMetric,
)
from encord_active.analysis.types import (
    AnnotationMetadata,
    MaskBatchTensor,
    MetricBatchResult,
)
from encord_active.db.metrics import MetricType


class MaximumLabelIOUMetric(ObjectByFrameMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_max_iou",
            long_name="Label Duplicates",
            desc="Assigns the maximum IOU between each label and the rest of the labels in the frame.",
            metric_type=MetricType.NORMAL,
        )

    def calculate(
        self,
        annotations: Dict[str, AnnotationMetadata],
        annotation_deps: dict[str, MetricDependencies],
    ) -> Dict[str, MetricResultOptionalAnnotation]:
        objs = [
            (annotation_hash, annotation.mask, annotation.bounding_box.type(torch.int32).tolist())
            for annotation_hash, annotation in annotations.items()
            if annotation.mask is not None and annotation.bounding_box is not None
        ]
        b = len(objs)
        if b == 0:
            return {}
        if b == 1:
            # skip 1 label case.
            return {obj_hash: 0.0 for obj_hash, obj_mask, obj_bb in objs}
        # Avoid calculating twice.
        calculated_iou: Dict[Tuple[str, str], float] = {}
        max_iou_results: Dict[str, MetricResultOptionalAnnotation] = {}
        for obj_hash1, obj_mask1, obj_bb1 in objs:
            obj_iou_values = []
            b1x1, b1y1, b1x2, b1y2 = obj_bb1
            for obj_hash2, obj_mask2, obj_bb2 in objs:
                if obj_hash1 == obj_hash2:
                    continue
                k = (obj_hash1, obj_hash2) if obj_hash1 < obj_hash2 else (obj_hash2, obj_hash1)
                if k in calculated_iou:
                    obj_iou_values.append(calculated_iou[k])
                    continue
                b2x1, b2y1, b2x2, b2y2 = obj_bb2
                # Quick bb-bb intersection test to avoid expensive calculation when not needed.
                if b2x1 > b1x2 or b2x2 < b1x1 or b2y1 > b1y2 or b2y2 < b1y1:
                    intersect = 0.0  # Quick bb-collision exclude
                else:
                    intersect = float((obj_mask1 & obj_mask2).sum().item())
                if intersect == 0.0:
                    iou = 0.0
                else:
                    iou = intersect / float((obj_mask1 | obj_mask2).sum().item())
                calculated_iou[k] = iou
                obj_iou_values.append(iou)
            max_iou_results[obj_hash1] = max(obj_iou_values, default=0.0)

        """GPU optimized style - not used as currently cpu is easier due to no batching
        masks = torch.stack([obj_mask for obj_hash, obj_mask in objs])
        intersections = (masks.unsqueeze(0) & masks.unsqueeze(1)).view(b, b, -1).sum(-1)
        unions = (masks.unsqueeze(0) | masks.unsqueeze(1)).view(b, b, -1).sum()
        ious = intersections / unions
        ious[np.diag_indices_from(ious)] = -1  # don't consider self against self
        max_ious = ious.max(1).values.cpu()
        return dict(zip([obj_hash for obj_hash, obj_mask in objs], max_ious))
        """
        return max_iou_results

    def calculate_batched(
        self,
        masks: MaskBatchTensor,  # FIXME: batch_extra_dimensions??
    ) -> MetricBatchResult:
        # intersections = (masks.unsqueeze(0) & masks.unsqueeze(1)).view(b, b, -1).sum(-1)
        # unions = (masks.unsqueeze(0) | masks.unsqueeze(1)).view(b, b, -1).sum()
        # ious = intersections / unions
        raise ValueError()
