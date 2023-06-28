import torch
from clip import load as clip_load
from torch import BoolTensor, ByteTensor, FloatTensor, IntTensor
from torch.types import Device

from encord_active.analysis.embedding import PureImageEmbedding


class ClipImgEmbedding(PureImageEmbedding):
    def __init__(self, device: Device, ident: str, model_name: str) -> None:
        super().__init__(
            ident=ident,
            dependencies=set(),
            allow_object_embedding=False,
            allow_queries=True,
        )
        self.model, self.preprocess = clip_load(name=model_name, device=device)
        self.device = device

    def evaluate_embedding(self, image: ByteTensor, mask: BoolTensor | None) -> FloatTensor:
        if mask is not None:
            image = torch.masked_fill(image, ~mask, 0)
        preprocessed = self.model.preprocess(image)
        return self.model.encode_image(preprocessed)
