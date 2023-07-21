from typing import Optional

import torch
from clip import load as clip_load
from torch import BoolTensor, ByteTensor, FloatTensor
from torch.types import Device
from torchvision.transforms import ToPILImage, InterpolationMode
from torchvision.transforms.v2 import CenterCrop, Normalize, Resize, Compose

from encord_active.analysis.embedding import PureImageEmbedding


class ClipImgEmbedding(PureImageEmbedding):
    def __init__(self, device: Device, ident: str, model_name: str) -> None:
        super().__init__(
            ident=ident,
            dependencies=set(),
            allow_object_embedding=True,
            allow_queries=True,
        )
        self.model, self.preprocess = clip_load(name=model_name, device=device)
        self.device = device
        # NOTE: device round trips are slow - hence execute everything on the tensor
        # this was extracted from the clip implementation
        # FIXME: this is for mps dev (BICUBIC is correct mode but not supported)
        interpolation = InterpolationMode.BICUBIC if self.device.type != "mps" else InterpolationMode.BILINEAR
        if interpolation != InterpolationMode.BICUBIC:
            import logging
            logging.warning(f"Using different InterpolationMode as currently executing under 'mps' torch backend.")
        n_px = self.model.visual.input_resolution
        self.preprocess = Compose([
            Resize(n_px, interpolation=interpolation, antialias=True),
            CenterCrop(n_px),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def evaluate_embedding(self, image: ByteTensor, mask: Optional[BoolTensor]) -> FloatTensor:
        if mask is not None:
            image = torch.masked_fill(image, ~mask, 0)
        preprocessed = torch.stack([self.preprocess(image.type(torch.float32))])
        return self.model.encode_image(preprocessed.to(self.device)).reshape(-1)
