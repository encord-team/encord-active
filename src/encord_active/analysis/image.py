from abc import ABC, abstractmethod
from typing import List

from torch import IntTensor
from numpy import nparray


class ImageContext(ABC):
    @abstractmethod
    def as_tensor(self) -> IntTensor:
        ...

    @abstractmethod
    def as_np_array(self) -> nparray:
        ...

    @abstractmethod
    def img_width(self) -> int:
        ...

    @abstractmethod
    def img_height(self) -> int:
        ...

    @abstractmethod
    def objects(self) -> List[object]:  # TODO!!!!
        ...

    def classifications(self) -> List[object]:  # TODO!!!
        ...
