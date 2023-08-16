from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Union

import torch

from encord_active.analysis.base import (
    BaseEvaluation,
    BaseFrameBatchInput,
    BaseFrameBatchOutput,
    BaseFrameInput,
    BaseFrameOutput,
)
from encord_active.analysis.types import HSVTensor, ImageTensor, MetricResult


class BaseConverter(BaseEvaluation, metaclass=ABCMeta):
    """
    Base Data Converter
    """

    def __init__(self, ident: str) -> None:
        super().__init__(ident)

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        """
        Base implementation of the raw_calculate method, this api should
        be considered unstable.
        """
        image = self.convert(frame.image)
        annotations: Dict[str, MetricResult] = {}
        for k, v in frame.annotations.items():
            if v.bounding_box is None or v.mask is None:
                annotations[k] = image
            else:
                x1, y1, x2, y2 = v.bounding_box.type(torch.int32).tolist()
                if len(image.shape) == 3:
                    annotation_image = image[:, y1 : y2 + 1, x1 : x2 + 1]
                    mask = v.mask[y1 : y2 + 1, x1 : x2 + 1]
                    annotations[k] = torch.masked_select(annotation_image, mask).reshape(3, -1)
                else:
                    annotation_image = image[y1 : y2 + 1, x1 : x2 + 1]
                    mask = v.mask[y1 : y2 + 1, x1 : x2 + 1]
                    annotations[k] = torch.masked_select(annotation_image, mask)
        return BaseFrameOutput(image=image, image_comment=None, annotations=annotations, annotation_comments={})

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        """
        Base implementation of batched metric calculation.
        """
        images = self.convert(frame.images)
        return BaseFrameBatchOutput(
            images=images,
            objects=None,
            classifications=None,
        )

    @abstractmethod
    def convert(self, image: ImageTensor) -> Union[ImageTensor, HSVTensor]:
        ...
