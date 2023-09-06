from typing import Union

from encord_active.analysis.conversion import BaseConverter
from encord_active.analysis.types import HSVTensor, ImageTensor
from encord_active.analysis.util import rgb_to_hsv

HSVRange = Union[tuple[int, int], list[tuple[int, int]]]


class RGBToHSV(BaseConverter):
    def __init__(self) -> None:
        super().__init__(
            ident="ephemeral_hsv_image",
        )

    def convert(self, image: ImageTensor) -> HSVTensor:
        """
        This will return a new image in HSV space. The range of values will be

        H: [0; 2pi], S: [0, 1], V: [0, 1]
        """

        return rgb_to_hsv(image)