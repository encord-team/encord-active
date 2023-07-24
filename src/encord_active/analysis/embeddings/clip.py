from typing import Optional

import torch
import torchvision.ops.boxes
from clip import load as clip_load
from torch import BoolTensor, ByteTensor, FloatTensor
from torch.types import Device
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import CenterCrop, Normalize, Resize, Compose
from torchvision.utils import make_grid

from encord_active.analysis.embedding import PureImageEmbedding, ImageEmbeddingResult
from encord_active.analysis.metric import ObjectOnlyBatchInput
from encord_active.analysis.types import ImageBatchTensor, EmbeddingBatchTensor, MaskBatchTensor
from encord_active.analysis.util.torch import batch_size


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
        # FIXME: mask is not implemented properly, require scale by bb & apply mask - see untested
        # batch implementation.
        if mask is not None:
            boxes = torchvision.ops.masks_to_boxes(mask.unsqueeze(0)).cpu().squeeze(0)
            x1, y1, x2, y2 = boxes.type(torch.int32).tolist()
            image = torch.masked_fill(image, ~mask, 0)
            image = image[:, x1:x2, y1:y2]
        preprocessed = self.preprocess(image.type(torch.float32)).unsqueeze(0)
        return self.model.encode_image(preprocessed.to(self.device)).reshape(-1)

    def evaluate_embedding_batched(
        self,
        image: ImageBatchTensor,
        objects: Optional[ObjectOnlyBatchInput]
    ) -> ImageEmbeddingResult:
        image_preprocessed = self.preprocess(image.type(torch.float32))
        image_embeddings = self.model.encode_image(image_preprocessed)
        object_embeddings = None
        if objects is not None:
            # First convert to resized image space
            object_images = torch.index_select(
                image,
                0,
                objects.objects_image_indices
            )

            # Generate bounding box grid, this is the size of the smallest dimension.
            n_px: int = self.model.visual.input_resolution
            x_grid = torch.arange(0, n_px).repeat(n_px, 1)
            y_grid = torch.arange(0, n_px).unsqueeze(1).repeat(1, n_px)
            xy_grid = torch.concat([x_grid.unsqueeze(2), y_grid.unsqueeze(2)], 2).type(torch.float32)
            xy_grid_batch = xy_grid.repeat((batch_size(object_images), 1, 1, 1))

            # Scale the grid by the bounding boxes.
            #  FIXME: select best scaling strategy for width & height
            #   current implementation is (no mask) maximum dimension.
            bounding_boxes = objects.objects_bounding_boxes
            xy1 = bounding_boxes[:, :2]
            xy2 = bounding_boxes[:, 2:]
            wh = xy2 - xy1
            scaling = torch.amax(wh, dim=1, keepdim=True) / float(n_px)
            midpoints = ((xy1 + xy2) / 2.0) - (scaling / 2.0)

            # (C * (scaling / n_px)) + (midpoint-(scaling / 2))
            object_bb_grid = (xy_grid_batch * scaling.unsqeeze(-1).unsqueeze(-1)) + midpoints.unsqueeze(1).unsqueeze(1)

            # Apply grid transforms to generate
            object_bb_images = torch.nn.functional.grid_sample(
                object_images,
                object_bb_grid,
            )
            # FIXME: apply masks as well
            object_preprocessed = self.preprocess(object_bb_images) # FIXME: resize twice?
            object_embeddings = self.model.encode_image(object_preprocessed)
        return ImageEmbeddingResult(
            images=image_embeddings,
            objects=object_embeddings
        )
