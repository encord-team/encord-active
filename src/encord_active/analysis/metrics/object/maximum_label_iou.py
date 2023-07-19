from typing import Dict

import numpy as np
import torch

from encord_active.analysis.metric import MetricDependencies, ObjectByFrameMetric
from encord_active.analysis.types import MetricResult, AnnotationMetadata


class MaximumLabelIOUMetric(ObjectByFrameMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="maximum",
            dependencies=set(),
            long_name="Object Count",
            desc="Assigns the maximum IOU between each label and the rest of the labels in the frame.",
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
        if b == 1:
            # skip 1 label case.
            return {
                obj_hash: 0.0
                for obj_hash, obj_mask in objs
            }
        masks = torch.stack([obj_mask for obj_hash, obj_mask in objs])
        intersections = (masks.unsqueeze(0) & masks.unsqueeze(1)).view(b, b, -1).sum(-1)
        unions = (masks.unsqueeze(0) | masks.unsqueeze(1)).view(b, b, -1).sum()
        ious = intersections / unions
        ious[np.diag_indices_from(ious)] = -1  # don't consider self against self
        max_ious = ious.max(1).values.cpu()
        return dict(zip([obj_hash for obj_hash, obj_mask in objs], max_ious))
