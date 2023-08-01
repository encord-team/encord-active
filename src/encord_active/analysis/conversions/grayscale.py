import torch

from encord_active.analysis.conversion import BaseConverter
from encord_active.analysis.types import HSVTensor, ImageTensor


class RGBToGray(BaseConverter):
    def __init__(self) -> None:
        super().__init__(
            ident="ephemeral_grayscale_image",
        )

    def convert(self, image: ImageTensor) -> HSVTensor:
        """
        This will return a new image in Grayscale.
        This reduces the 3 channels to 1 - removing a dimension
        """
        return torch.mean(image, dim=-3, dtype=torch.float32).type(torch.uint8)
