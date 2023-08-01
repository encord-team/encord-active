from typing import Optional

import cv2
import torch
from torch import FloatTensor, BoolTensor

from encord_active.analysis.embedding import PureObjectEmbedding
from encord_active.analysis.types import MaskTensor, EmbeddingTensor
from encord_active.analysis.util import image_width, image_height


class HuMomentEmbeddings(PureObjectEmbedding):
    def __init__(self) -> None:
        super().__init__("embedding_hu", set())

    def evaluate_embedding(self, mask: MaskTensor) -> EmbeddingTensor:
        np_mask = (mask.cpu().type(torch.uint8) * 255).numpy()
        hu_moments = cv2.HuMoments(cv2.moments(np_mask)).flatten()
        return torch.from_numpy(hu_moments).type(torch.float32)

    def classification_embedding(self) -> Optional[EmbeddingTensor]:
        return torch.zeros(7, dtype=torch.float32)  # FIXME: return hardcoded whole image 'embedding'

    def wip_torch_hu_moment(self, mask: BoolTensor, bb: torch.Tensor) -> FloatTensor:
        bounding_boxes = bb
        xy1 = bounding_boxes[:, :2]
        xy2 = bounding_boxes[:, 2:]
        midpoints = (xy1 + xy2) / 2.0
        x_midpoint = midpoints[:, 0]
        y_midpoint = midpoints[:, 1]
        x_coordinates, y_coordinates = torch.meshgrid(
            torch.arange(0, image_width(mask), dtype=torch.float32),
            torch.arange(0, image_height(mask), dtype=torch.float32),
            indexing='ij'
        )
        x_centroid = x_coordinates - x_midpoint.unsqueeze(1).unsqueeze(2)
        y_centroid = y_coordinates - y_midpoint.unsqueeze(1).unsqueeze(2)
        x_moment = torch.masked_fill(x_centroid, ~mask, 0.0)
        y_moment = torch.masked_fill(y_centroid, ~mask, 0.0)

        # u_pq = torch.sum(x ** p * y ** q)
        x2_moment = torch.square(x_moment)
        x3_moment = x2_moment * x_moment
        y2_moment = torch.square(y_moment)
        y3_moment = y2_moment * y_moment

        # Torch implementation of hu-moments.
        u20 = mask
        u02 = mask
        u30 = mask
        u03 = mask
        u12 = mask
        u21 = mask
        u11 = mask

        # https://cvexplained.wordpress.com/2020/07/21/10-4-hu-moments/
        m1 = u20 + u02
        m2 = torch.square(u30 - u02) + (4 * torch.square(u11))
        m3 = torch.square(u30 - (3 * u12)) + torch.square((3 * u21) - u30)
        m4 = torch.square(u30 + u12) + torch.square(u21 + u03)
        m5 = (
            (u30 - (3 * u12)) * (u30 + u12) * (
                 torch.square() - 3*torch.square()
            )
        ) + (

        )

        return torch.stack([m1, m2, m3, m4, m5, m6, m7])
