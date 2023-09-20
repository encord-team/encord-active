import math
from typing import List, Optional, Union

import cv2
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
    # FIXME: unsure how much we gain scaling to pixel-space here as we need to add correct rounding logic anyway
    #  only user of scaled-to-pixel space is mask creation (which in-turn creates bounding boxes).
    width = float(img_w)  # FIXME: normalise or not to normalise??
    height = float(img_h)
    if annotation_type == AnnotationType.CLASSIFICATION:
        return None
    if annotation_type == AnnotationType.POLYGON or annotation_type == AnnotationType.POLYLINE:
        points = obj["polygon"] if annotation_type == AnnotationType.POLYGON else obj["polyline"]
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
    elif annotation_type == AnnotationType.ROTATABLE_BOUNDING_BOX:
        rbb = obj.get("rot_bounding_box", None) or obj.get("rotatableBoundingBox", None)
        x = rbb["x"]
        y = rbb["y"]
        w = rbb["w"]
        h = rbb["h"]
        x2 = x + w
        y2 = y + h
        theta = rbb["theta"]
        c = math.cos(theta)
        s = math.sin(theta)

        def _rotate(xv: float, yv: float) -> List[float]:
            xr = xv * c - yv * s
            yr = xv * s + yv * c
            return [xr, yr]

        data = [
            _rotate(x, y),
            _rotate(x, y2),
            _rotate(x2, y2),
            _rotate(x2, y),
        ]
        return torch.tensor(data, dtype=torch.float32)
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
    device: Device, annotation_type: AnnotationType, obj: dict, img_w: int, img_h: int, points: Optional[PointTensor]
) -> Optional[MaskTensor]:
    # TODO fix me
    if annotation_type == AnnotationType.CLASSIFICATION:
        return None
    elif annotation_type == AnnotationType.BITMASK:
        from pycocotools import mask

        bitmask_obj = obj["bitmask"]
        rle_string = bitmask_obj["rleString"]
        rle_width = bitmask_obj["width"]
        rle_height = bitmask_obj["height"]
        bitmask: np.ndarray = mask.decode(
            {
                "counts": rle_string,
                "size": [rle_width, rle_height],
            }
        )
        rle_y = bitmask_obj["top"]
        rle_x = bitmask_obj["left"]
        if rle_x != 0 or rle_y != 0:
            raise ValueError("Not supported")
        if tuple(bitmask.shape) != (img_h, img_w):
            raise RuntimeError(
                f"Bugged bitmask decode, shape does not match: {tuple(bitmask.shape)} != {(img_h, img_w)}"
            )
        tensor = torch.from_numpy(bitmask).type(torch.bool)
        return tensor.T  # Convert to height, width format
    elif points is None:
        raise ValueError("BUG: points value not passed as argument for annotation type containing points")
    elif annotation_type == AnnotationType.POLYGON:
        return polygon_mask(points, img_w, img_h)
    elif annotation_type == AnnotationType.POLYLINE:
        return polyline_mask(points, img_w, img_h)
    elif annotation_type == AnnotationType.BOUNDING_BOX:
        # FIXME: special case annotation bounding box creation for this to avoid the redundant calculation!!
        # FIXME: cleanup and keep in gpu memory
        array = points.cpu().numpy().tolist()
        return bounding_box_mask(
            device, Point(*[int(round(x)) for x in array[0]]), Point(*[int(round(x)) for x in array[2]]), img_w, img_h
        )
    elif annotation_type == AnnotationType.ROTATABLE_BOUNDING_BOX:
        return polygon_mask(points, img_w, img_h)
    elif annotation_type == AnnotationType.POINT:
        x, y = points.cpu().numpy().tolist()[0]
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
    channel_count = image.shape[-3]
    kernel = _laplacian_kernel.repeat(channel_count, 1, 1, 1).to(image.device)
    return conv2d(channels_as_batch, weight=kernel, padding=1, groups=channel_count)


def polyline_mask(coordinates: PointTensor, width: int, height: int) -> MaskTensor:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = coordinates.cpu().numpy().round(0).astype(np.int32)
    cv2.polylines(mask, [points], False, 1)
    for x, y in points:
        mask[min(y, height - 1), min(x, width - 1)] = 1
    return torch.tensor(mask).bool()


def polygon_mask(coordinates: PointTensor, width: int, height: int) -> MaskTensor:
    # FIXME: implement & verify rounding behaviour & interaction.
    # TODO: Implement the winding algorithm in torch instead for performance
    mask = np.zeros((height, width), dtype=np.uint8)
    points = coordinates.cpu().numpy().round(0).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    for x, y in points:
        mask[min(y, height - 1), min(x, width - 1)] = 1
    return torch.tensor(mask).bool()


def bounding_box_mask(device: Device, top_left: Point, bottom_right: Point, width: int, height: int) -> MaskTensor:
    # FIXME: verify rounding behaviour & interaction.
    mask = torch.zeros(height, width, dtype=torch.bool, device=device).bool()
    mask[
        min(top_left.y, height - 1) : min(bottom_right.y, height - 1),
        min(top_left.x, width - 1) : min(bottom_right.x, width - 1),
    ] = True
    return mask


def point_mask(device: Device, x: float, y: float, width: int, height: int) -> MaskTensor:
    # FIXME: verify rounding behaviour & interaction.
    mask = torch.zeros(height, width, dtype=torch.bool, device=device).bool()
    x_i = round(x * width)
    y_i = round(y * height)
    mask[min(y_i, height - 1), min(x_i, width - 1)] = True
    return mask  # type: ignore


def image_width(image: Union[ImageTensor, MaskTensor, ImageBatchTensor, MaskBatchTensor]) -> int:
    return image.shape[-1]


def image_height(image: Union[ImageTensor, MaskTensor, ImageBatchTensor, MaskBatchTensor]) -> int:
    return image.shape[-2]


def batch_size(batch: Union[ImageBatchTensor, MaskBatchTensor]) -> int:
    return batch.shape[0]
