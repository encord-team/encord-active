from encord_active.analysis.conversion import BaseConverter
from encord_active.analysis.types import HSVTensor, ImageTensor
from encord_active.analysis.util import laplacian2d


class RGBToLaplacian(BaseConverter):
    def __init__(self) -> None:
        super().__init__(
            ident="ephemeral_laplacian_image",
        )

    def convert(self, image: ImageTensor) -> HSVTensor:
        """
        This will return a new image with the laplacian kernel applied
        """
        return laplacian2d(image.unsqueeze(0)).squeeze(0)
