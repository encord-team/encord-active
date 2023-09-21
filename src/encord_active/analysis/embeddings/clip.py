from typing import Dict, Optional

import torch
import torchvision.ops.boxes
from clip import load as clip_load
from torchvision.transforms import InterpolationMode

torchvision.disable_beta_transforms_warning()
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from encord_active.analysis.base import BaseFrameInput, BaseFrameOutput
from encord_active.analysis.embedding import ImageEmbeddingResult, PureImageEmbedding
from encord_active.analysis.metric import ObjectOnlyBatchInput
from encord_active.analysis.types import (
    EmbeddingTensor,
    ImageBatchTensor,
    ImageTensor,
    MaskTensor,
    MetricResult,
)
from encord_active.analysis.util.torch import batch_size


class ClipImgEmbedding(PureImageEmbedding):
    def __init__(self, device: torch.device, ident: str, model_name: str) -> None:
        super().__init__(
            ident=ident,
            allow_object_embedding=True,
            allow_queries=True,
        )
        self.model, self.preprocess = clip_load(name=model_name, device=device)
        self.device = device
        # NOTE: device round trips are slow - hence execute everything on the tensor
        # this was extracted from the clip implementation
        n_px = self.model.visual.input_resolution
        self.preprocess = Compose(
            [
                Resize(n_px, interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop(n_px),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        annotation_embeddings: Dict[str, MetricResult] = {}
        # NOTE: override implementation to batch image & all objects together!!
        pre_device = self.device if self.device.type != "mps" else torch.device("cpu")
        embedding_stack = [self.preprocess(frame.image.type(torch.float32).to(pre_device))]
        annotation_ordered = list(frame.annotations.items())
        for _, annotation in annotation_ordered:
            if annotation.mask is None or annotation.bounding_box is None:
                # classification
                continue
            x1, y1, x2, y2 = annotation.bounding_box.type(torch.int32).tolist()
            frame_image = frame.image.to(pre_device)
            annotation_mask = annotation.mask.to(pre_device)
            annotate_image = torch.masked_fill(
                frame_image[:, y1 : y2 + 1, x1 : x2 + 1], ~annotation_mask[y1 : y2 + 1, x1 : x2 + 1], 0
            )
            embedding_stack.append(self.preprocess(annotate_image.type(torch.float32)))

        embedding_encode_stack = torch.stack(embedding_stack).to(self.device)
        embedding_result_stack = self.model.encode_image(embedding_encode_stack).to(self.device)
        image_embedding = embedding_result_stack[0]
        idx = 1
        for annotation_hash, annotation in annotation_ordered:
            if annotation.mask is None or annotation.bounding_box is None:
                # classification
                annotation_embeddings[annotation_hash] = image_embedding
            else:
                annotation_embeddings[annotation_hash] = embedding_result_stack[idx].cpu()
                idx += 1
        return BaseFrameOutput(
            image=image_embedding,
            image_comment=None,
            annotations=annotation_embeddings,
            annotation_comments={},
        )

    def evaluate_embedding(self, image: ImageTensor, mask: Optional[MaskTensor]) -> EmbeddingTensor:
        # FIXME: mask is not implemented properly, require scale by bb & apply mask - see untested
        # batch implementation.
        print("BUG: old eval embedding run")
        if mask is not None:
            boxes = torchvision.ops.masks_to_boxes(mask.unsqueeze(0)).cpu().squeeze(0)
            x1, y1, x2, y2 = boxes.type(torch.int32).tolist()
            image = torch.masked_fill(image, ~mask, 0)
            image = image[:, y1 : y2 + 1, x1 : x2 + 1]
        try:
            preprocessed = self.preprocess(image.type(torch.float32)).unsqueeze(0)
        except Exception:
            print(
                f"DEBUG: {image.shape} / {None if mask is None else mask.shape} "
                f"/ {None if mask is None else torch.min(mask).item()} "
                f"/ {None if mask is None else torch.max(mask).item()}"
            )
            raise
        return self.model.encode_image(preprocessed.to(self.device)).reshape(-1)

    def evaluate_embedding_batched(
        self, image: ImageBatchTensor, objects: Optional[ObjectOnlyBatchInput]
    ) -> ImageEmbeddingResult:
        image_preprocessed = self.preprocess(image.type(torch.float32))
        image_embeddings = self.model.encode_image(image_preprocessed)
        object_embeddings = None
        if objects is not None:
            # First convert to resized image space
            object_images = torch.index_select(image, 0, objects.objects_image_indices)

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
            object_bb_grid = (xy_grid_batch * scaling.unsqueeze(-1).unsqueeze(-1)) + midpoints.unsqueeze(1).unsqueeze(1)

            # Apply grid transforms to generate
            object_bb_images = torch.nn.functional.grid_sample(
                object_images,
                object_bb_grid,
            )
            # FIXME: apply masks as well
            object_preprocessed = self.preprocess(object_bb_images)  # FIXME: resize twice?
            object_embeddings = self.model.encode_image(object_preprocessed)
        return ImageEmbeddingResult(images=image_embeddings, objects=object_embeddings)
