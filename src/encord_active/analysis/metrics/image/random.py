import torch
from typing import Optional

from encord_active.analysis.base import BaseFrameAnnotationBatchInput, BaseFrameBatchInput, BaseFrameBatchOutput
from encord_active.analysis.metric import MetricDependencies, OneImageMetric, ImageObjectOnlyOutputBatch, \
    ObjectOnlyBatchInput
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult, MetricBatchDependencies, \
    ImageBatchTensor, MaskBatchTensor, MetricBatchResult
from encord_active.analysis.util.torch import batch_size


class RandomMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident="metric_random",
            dependencies=set(),
            long_name="Random Value",
            desc="Assigns random float value in the range [0; 1].",
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
            objects=None if frame.annotations is None else torch.rand(
                batch_size(frame.annotations.objects_image_indices)
            ),
            classifications=None if frame.annotations is None else torch.rand(
                batch_size(frame.annotations.classifications_image_indices)
            ),
        )

    def calculate_batched(
        self,
        deps: MetricBatchDependencies,
        image: ImageBatchTensor,
        annotation: Optional[ObjectOnlyBatchInput]
    ) -> ImageObjectOnlyOutputBatch:
        raise ValueError(f"Base implementation override")
