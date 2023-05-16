from typing import Tuple

from clip import load as clip_load
from torch import FloatTensor, flatten
from torch.nn import Module
from torch.types import Device

from encord_active.analysis.context import ImageEvalContext
from encord_active.analysis.embedding import ImageEmbedding
from encord_active.analysis.image import ImageContext


class CnnImgEmbedding(ImageEmbedding):
    def __init__(self, device: Device) -> None:
        super().__init__(
            ident='clip-image-embedding',
            dependencies=set(),
            long_name='Clip Embedding',
            short_desc='TODO',
            long_desc='TODO',
            allow_nearby_query=True,
        )
        self.model, self.preprocess = clip_load("ViT-B/32", device=device)
        self.device = device

    def calc_embedding(self, context: ImageEvalContext, image: ImageContext) -> FloatTensor:
        image_tensor = image.as_tensor()
        image_preprocessed = self.model.preprocess(image_tensor)
        return self.model.encode_image(image_preprocessed)

# TODO: CNN Label + Classification embedding?!
