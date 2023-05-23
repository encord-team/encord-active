import torch
from torch import FloatTensor, BoolTensor, ByteTensor
from kornia.utils import draw_convex_polygon


def rgb_to_hsv(image: ByteTensor) -> ByteTensor:
    return image  # FIXME: implement


def draw_polygon_mask(width: int, height: int, coordinates: FloatTensor, invert: bool = False) -> BoolTensor:
    zero_mask = torch.zeros(width, height, dtype=torch.bool).bool()
    draw_convex_polygon(zero_mask, coordinates, torch.ones(1, dtype=torch.bool))
    return zero_mask

# FIXME: batch all processing, Object & Image batches potentially (Object batches most likely to be useful).
# But potentially look into doing both?!

def draw_bbox_mask(width: int, height: int, bounds: FloatTensor) -> BoolTensor:
    ...