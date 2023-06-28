from abc import ABCMeta, abstractmethod

from encord_active.analysis.base import BaseEvaluation
from encord_active.analysis.types import HSVTensor, ImageTensor


class BaseConverter(BaseEvaluation, metaclass=ABCMeta):
    """
    Base Data Converter
    """

    def __init__(self, ident: str, dependencies: set[str]) -> None:
        super().__init__(ident, dependencies)

    @abstractmethod
    def convert(self, image: ImageTensor) -> ImageTensor | HSVTensor:
        ...
