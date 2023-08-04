from abc import ABCMeta, abstractmethod
from typing import Optional, Union

from encord_active.analysis.base import (
    BaseEvaluation,
    BaseFrameBatchInput,
    BaseFrameBatchOutput,
    BaseFrameInput,
    BaseFrameOutput,
)
from encord_active.analysis.types import HSVTensor, ImageTensor


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
        return BaseFrameOutput(
            image=image,
            annotations={k: image for k in frame.annotations.keys()},
        )

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
