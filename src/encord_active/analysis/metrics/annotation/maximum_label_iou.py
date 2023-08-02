from typing import Dict

import numpy as np
import torch

from encord_active.analysis.metric import MetricDependencies, ObjectByFrameMetric
from encord_active.analysis.types import (
    AnnotationMetadata,
    MaskBatchTensor,
    MetricBatchResult,
    MetricResult,
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
    ) -> dict[str, MetricResult]:
        objs = [
            (annotation_hash, annotation.mask)
            for annotation_hash, annotation in annotations.items()
            if annotation.mask is not None
        ]
        b = len(objs)
        if b == 0:
            return {}
        if b == 1:
            # skip 1 label case.
            return {obj_hash: 0.0 for obj_hash, obj_mask in objs}
        masks = torch.stack([obj_mask for obj_hash, obj_mask in objs])
        intersections = (masks.unsqueeze(0) & masks.unsqueeze(1)).view(b, b, -1).sum(-1)
        unions = (masks.unsqueeze(0) | masks.unsqueeze(1)).view(b, b, -1).sum()
        ious = intersections / unions
        ious[np.diag_indices_from(ious)] = -1  # don't consider self against self
        max_ious = ious.max(1).values.cpu()
        return dict(zip([obj_hash for obj_hash, obj_mask in objs], max_ious))

    def calculate_batched(
        self,
        masks: MaskBatchTensor,  # FIXME: batch_extra_dimensions??
    ) -> MetricBatchResult:
        # intersections = (masks.unsqueeze(0) & masks.unsqueeze(1)).view(b, b, -1).sum(-1)
        # unions = (masks.unsqueeze(0) | masks.unsqueeze(1)).view(b, b, -1).sum()
        # ious = intersections / unions
        raise ValueError()
