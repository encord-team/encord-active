import torch
from torch import IntTensor, FloatTensor, BoolTensor, from_numpy
from torch.nn.functional import conv2d
from typing import List
from PIL import Image
import numpy as np

_laplacian_kernel: List[List[float]] = [
    [0.0, 1.0, 0.0],
    [1.0, -4.0, 1.0],
    [0.0, 1.0, 0.0]
]


def pillow_to_tensor(image: Image.Image) -> IntTensor:
    pass


def tensor_to_pillow(tensor: IntTensor) -> Image.Image:
    pass


def laplacian2d(image: IntTensor) -> FloatTensor:
    return conv2d(image.float(), weight=from_numpy(np.array(_laplacian_kernel))).float()


def polygon_mask(coordinates: List[(int, int)], width: int, height: int) -> BoolTensor:
    mask = torch.zeros(width, height, dtype=torch.bool)
    # Implement the winding algorithm
    for x, y in coordinates:
        miny = min(y)
        maxy = max(y)


def bounding_box_mask(min_coord: (float, float), max_coord: (float, float), width: int, height: int) -> BoolTensor:
    mask = torch.zeros(width, height, dtype=torch.bool).bool()

    return mask


def point_mask(x: float, y: float, width: int, height: int) -> BoolTensor:
    mask = torch.zeros(width, height, dtype=torch.bool).bool()
    x_i = round(x * width)
    y_i = round(y * height)
    mask[x_i, y_i] = True
    return mask


def polygon_iou(coordinates: FloatTensor, width: int, height: int) -> BoolTensor:
    pass
