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
    MetricKey,
    ObjectMetadata,
    Point,
    PointTensor,
)
from encord_active.lib.labels.object import ObjectShape

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


def _obj_to_points(obj: dict, img_w: int, img_h: int) -> PointTensor:
    # TODO fix me
    return torch.zeros((10, 2), dtype=torch.float)


def _obj_to_mask(obj: dict, img_w: int, img_h) -> MaskTensor:
    # TODO fix me
    shape = obj.get("shape")
    if not shape:
        raise ValueError("Object has no `shape` property")
    if shape == ObjectShape.POLYGON:
        ...
    elif shape == ObjectShape.POLYLINE:
        ...
    elif shape == ObjectShape.BOUNDING_BOX:
        ...
    elif shape == ObjectShape.ROTATABLE_BOUNDING_BOX:
        ...
    elif shape == ObjectShape.KEY_POINT:
        ...
    elif shape == ObjectShape.SKELETON:
        ...
    return torch.zeros((img_h, img_w), dtype=torch.uint8)


def obj_to_object_meta(obj: dict, img_w: int, img_h: int) -> ObjectMetadata:
    points = _obj_to_points(obj, img_w=img_w, img_h=img_h)
    mask = _obj_to_mask(obj, img_w=img_w, img_h=img_h)
    return ObjectMetadata(feature_hash=obj["featureHash"], object_hash=obj["objectHash"], points=points, mask=mask)


def data_unit_to_object_meta(data_unit: dict, partial_key: MetricKey) -> dict[MetricKey, ObjectMetadata]:
    img_w, img_h = data_unit.get("width"), data_unit.get("height")
    if img_w is None or img_h is None:
        raise ValueError("The data unit does not provide image shape so can't construct object metadata")

    obj_list = data_unit.get("labels", {}).get("objects", [])
    if not obj_list:
        return {}

    return {
        partial_key.with_annotation(o["objectHash"]): obj_to_object_meta(o, img_w=img_w, img_h=img_h) for o in obj_list
    }


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
