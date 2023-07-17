from abc import ABCMeta, abstractmethod
from typing import Optional

from encord_active.analysis.base import BaseEvaluation, BaseFrameOutput, BaseFrameInput
from encord_active.analysis.types import HSVTensor, ImageTensor


class BaseConverter(BaseEvaluation, metaclass=ABCMeta):
    """
    Base Data Converter
    """

    def __init__(self, ident: str, dependencies: set[str]) -> None:
        super().__init__(ident, dependencies)

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
        return BaseFrameOutput(
            image=self.convert(frame.image),
            annotations={},
        )

    @abstractmethod
    def convert(self, image: ImageTensor) -> ImageTensor | HSVTensor:
        ...
