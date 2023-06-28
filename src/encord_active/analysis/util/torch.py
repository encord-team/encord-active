import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import conv2d
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from encord_active.analysis.types import (
    ImageTensor,
    LaplacianTensor,
    MaskTensor,
    Point,
    PointTensor,
)

_laplacian_kernel = (
    torch.tensor([0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0], requires_grad=False).reshape(1, 1, 3, 3).float()
)


def pillow_to_tensor(image: Image.Image) -> ImageTensor:
    """
    Returns an `uint8` RGB tensor of shape `[3, height, width]`.
    """
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return pil_to_tensor(image)


def tensor_to_pillow(tensor: ImageTensor) -> Image.Image:
    return to_pil_image(tensor.detach().cpu().numpy())


def laplacian2d(image: ImageTensor) -> LaplacianTensor:
    """
    Applies Laplacian kernel over Image
    """
    channels_as_batch = image.float().unsqueeze(1)  # [c, 1, h, w]
    return conv2d(channels_as_batch, weight=_laplacian_kernel).float().unsqueeze(1)  # type: ignore


def polygon_mask(coordinates: PointTensor, width: int, height: int) -> MaskTensor:
    # TODO: Implement the winding algorithm in torch instead for performance
    mask = np.zeros((width, height), dtype=np.uint8)
    points = coordinates.numpy().round(0).astype(np.uint8).reshape(-1, 1, 2)
    mask = cv2.fillPoly(mask, points, 1)
    return torch.tensor(mask).bool()


def bounding_box_mask(top_left: Point, bottom_right: Point, width: int, height: int) -> MaskTensor:
    mask = torch.zeros(width, height, dtype=torch.bool).bool()
    mask[top_left.y : bottom_right.y, top_left.x : bottom_right.x] = True
    return mask


def point_mask(x: float, y: float, width: int, height: int) -> MaskTensor:
    mask = torch.zeros(width, height, dtype=torch.bool).bool()
    x_i = round(x * width)
    y_i = round(y * height)
    mask[x_i, y_i] = True
    return mask  # type: ignore


def mask_to_box_extremes(mask: MaskTensor) -> tuple[Point, Point]:
    """
    Returns top_left and bottom_right point that includes `True` values.
    Make sure to not miss +1 index errors here. If you want to crop an
    image, for example, you would do
    ```
    tl, br = mask_box(mask)
    crop = img[:, tl.y:br.y+1, tl.x:br.x+1]
    ```
    """
    yx_coords = torch.stack(torch.where(mask)).T  # [n, 2]

    y_min, x_min = torch.min(yx_coords, 0)[0]
    y_max, x_max = torch.max(yx_coords, 0)[0]

    return Point(x_min, y_min), Point(x_max, y_max)


def image_width(image: ImageTensor | MaskTensor) -> int:
    return image.shape[-1]


def image_height(image: ImageTensor) -> int:
    return image.shape[-2]
