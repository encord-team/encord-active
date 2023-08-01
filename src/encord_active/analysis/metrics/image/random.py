from typing import Optional

import torch

from encord_active.analysis.base import BaseFrameBatchInput, BaseFrameBatchOutput
from encord_active.analysis.metric import (
    ImageObjectOnlyOutputBatch,
    MetricDependencies,
    ObjectOnlyBatchInput,
    OneImageMetric,
)
from encord_active.analysis.types import (
    ImageBatchTensor,
    ImageTensor,
    MaskTensor,
    MetricBatchDependencies,
    MetricResult,
)
from encord_active.analysis.util.torch import batch_size
from encord_active.db.metrics import MetricType


class RandomMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_random",
            long_name="Random Value",
            desc="Assigns random float value in the range [0; 1].",
            metric_type=MetricType.NORMAL,
        )

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: Optional[MaskTensor]) -> MetricResult:
        return torch.rand(1)

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

    def calculate_batched(
        self, deps: MetricBatchDependencies, image: ImageBatchTensor, annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        raise ValueError(f"Base implementation override")
