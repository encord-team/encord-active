from typing import Optional

import torch

from encord_active.analysis.base import (
    BaseFrameBatchInput,
    BaseFrameBatchOutput,
    BaseFrameInput,
    BaseFrameOutput,
    BaseMetric,
)
from encord_active.analysis.util.torch import batch_size
from encord_active.db.metrics import MetricType


class RandomMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_random",
            long_name="Random Value",
            desc="Assigns random float value in the range [0; 1].",
            metric_type=MetricType.NORMAL,
        )

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        return BaseFrameOutput(
            image=torch.rand(1, device="cpu"),
            annotations={annotation_hash: torch.rand(1, device="cpu") for annotation_hash in frame.annotations.keys()},
        )

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        # Override implementation:
        #  ensure that classification values are independently random
        return BaseFrameBatchOutput(
            images=torch.rand(batch_size(frame.images)),
            objects=None
            if frame.annotations is None
            else torch.rand(batch_size(frame.annotations.objects_image_indices)),
            classifications=None
            if frame.annotations is None
            else torch.rand(batch_size(frame.annotations.classifications_image_indices)),
        )
