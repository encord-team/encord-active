import numpy as np
import torch

from encord_active.analysis.metric import MetricDependencies, ObjectByFrameMetric
from encord_active.analysis.types import MetricKey, MetricResult, ObjectMetadata


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
        img_deps: MetricDependencies,
        obj_deps: dict[str, MetricDependencies],
        objs: dict[MetricKey, ObjectMetadata],
    ) -> dict[MetricKey, MetricResult]:
        # TODO: instead of every object holding it's mask tensor, we could do it
        # with obj_deps if necessary. I do, however, think that it's such a common
        # thing that it makes sense to just have on the object.
        b = len(objs)
        masks = torch.stack([obj.mask for obj in objs.values()])
        intersections = (masks.unsqueeze(0) & masks.unsqueeze(1)).view(b, b, -1).sum(-1)
        unions = (masks.unsqueeze(0) | masks.unsqueeze(1)).view(b, b, -1).sum()
        ious = intersections / unions
        ious[np.diag_indices_from(ious)] = -1  # don't consider self against self
        max_ious = ious.max(1).values
        return dict(zip(objs.keys(), max_ious))
