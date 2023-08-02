from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import device as Device
from torch.nn.functional import conv2d
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from encord_active.analysis.types import (
    ImageBatchTensor,
    ImageTensor,
    LaplacianTensor,
    MaskBatchTensor,
    MaskTensor,
    Point,
    PointTensor,
)
from encord_active.db.enums import AnnotationType

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


def obj_to_points(annotation_type: AnnotationType, obj: dict, img_w: int, img_h: int) -> Optional[PointTensor]:
    width = float(img_w)  # FIXME: normalise or not to normalise??
    height = float(img_h)
    if annotation_type == AnnotationType.CLASSIFICATION:
        return None
    if annotation_type == AnnotationType.POLYGON or annotation_type == AnnotationType.POLYLINE:
        points = obj["polygon"]
        data = [[points[str(i)]["x"] * width, points[str(i)]["y"] * height] for i in range(len(points))]
        if len(data) == 0:
            raise ValueError(f"Polygon found with 0 points: {obj}")
        return torch.tensor(data, dtype=torch.float32)
    elif annotation_type == AnnotationType.BOUNDING_BOX:
        bb = obj.get("bounding_box", None) or obj.get("boundingBox", None)
        if bb is None:
            raise ValueError(f"Bounding box dict missing dimensions: {obj}")
        x = bb["x"]
        y = bb["y"]
        w = bb["w"]
        h = bb["h"]
        x2 = x + w
        y2 = y + h
        data = [
            [x * width, y * height],
            [x2 * width, y * height],
            [x2 * width, y2 * height],
            [x * width, y2 * height],
        ]
        return torch.tensor(data, dtype=torch.float32)
    elif annotation_type == AnnotationType.ROT_BOUNDING_BOX:
        raise ValueError("Unsupported annotation type rot bounding box")
    elif annotation_type == AnnotationType.POINT:
        point = obj["point"]["0"]
        x = point["x"]
        y = point["y"]
        data = [[x * width, y * height]]
        return torch.tensor(data, dtype=torch.float32)
    elif annotation_type == AnnotationType.SKELETON:
        raise ValueError("Skeleton object type is not supported")
    elif annotation_type == AnnotationType.BITMASK:
        return None
    raise ValueError(f"Unknown annotation type is not supported: {annotation_type}")


def obj_to_mask(
    device: Device, annotation_type: AnnotationType, img_w: int, img_h: int, points: Optional[PointTensor]
) -> Optional[MaskTensor]:
    # TODO fix me
    if annotation_type == AnnotationType.CLASSIFICATION:
        return None
    elif annotation_type == AnnotationType.BITMASK:
        raise ValueError("Bitmask object shape is not supported")
    elif points is None:
        raise ValueError("BUG: points value not passed as argument for annotation type containing points")
    elif annotation_type == AnnotationType.POLYGON:
        return polygon_mask(points, img_w, img_h)
    elif annotation_type == AnnotationType.POLYLINE:
        raise ValueError("Poly-line shape is not supported")
    elif annotation_type == AnnotationType.BOUNDING_BOX:
        # FIXME: cleanup and keep in gpu memory
        array = points.cpu().numpy().tolist()
        return bounding_box_mask(
            device, Point(*[int(round(x)) for x in array[0]]), Point(*[int(round(x)) for x in array[2]]), img_w, img_h
        )
    elif annotation_type == AnnotationType.ROT_BOUNDING_BOX:
        raise ValueError("Rotated bounding box shape is not supported")
    elif annotation_type == AnnotationType.POINT:
        x, y = points.cpu().numpy().tolist()
        return point_mask(device, x, y, img_w, img_h)
    elif annotation_type == AnnotationType.SKELETON:
        raise ValueError("Skeleton object shape is not supported")
    raise ValueError(f"Unknown annotation type is not supported: {annotation_type}")


def tensor_to_pillow(tensor: ImageTensor) -> Image.Image:
    return to_pil_image(tensor.detach().cpu().numpy())


def laplacian2d(image: ImageTensor) -> LaplacianTensor:
    """
    Applies Laplacian kernel over Image
    """
    channels_as_batch = image.float()
    return conv2d(channels_as_batch, weight=_laplacian_kernel, padding=1)


def polygon_mask(coordinates: PointTensor, width: int, height: int) -> MaskTensor:
    # TODO: Implement the winding algorithm in torch instead for performance
    mask = np.zeros((height, width), dtype=np.uint8)
    points = coordinates.cpu().numpy().round(0).astype(np.int32)
    # FIXME: broken!!! cv2.fillPoly(mask, [points], 1)
    for x, y in points:
        mask[min(y, height - 1), min(x, width - 1)] = 1
    # mask = cv2.fillPoly(mask, points, 1)
    # FIXME: this is broken!!
    return torch.tensor(mask).bool()


def bounding_box_mask(device: Device, top_left: Point, bottom_right: Point, width: int, height: int) -> MaskTensor:
    mask = torch.zeros(height, width, dtype=torch.bool, device=device).bool()
    mask[
        min(top_left.y, height - 1) : min(bottom_right.y, height - 1),
        min(top_left.x, width - 1) : min(bottom_right.x, width - 1),
    ] = True
    return mask


def point_mask(device: Device, x: float, y: float, width: int, height: int) -> MaskTensor:
    mask = torch.zeros(height, width, dtype=torch.bool, device=device).bool()
    x_i = round(x * width)
    y_i = round(y * height)
    mask[min(y_i, height - 1), min(x_i, width - 1)] = True
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
    # FIXME: cast to support mps backend - check perf & maybe remove in future
    yx_coords = yx_coords.type(torch.int32)
    y_min, x_min = torch.min(yx_coords, 0).values
    y_max, x_max = torch.max(yx_coords, 0).values

    # FIXME: deprecate with torchvision masks_to_boxes
    return Point(x_min, y_min), Point(x_max, y_max)


def image_width(image: Union[ImageTensor, MaskTensor, ImageBatchTensor, MaskBatchTensor]) -> int:
    return image.shape[-1]


def image_height(image: Union[ImageTensor, MaskTensor, ImageBatchTensor, MaskBatchTensor]) -> int:
    return image.shape[-2]


def batch_size(batch: Union[ImageBatchTensor, MaskBatchTensor]) -> int:
    return batch.shape[0]
