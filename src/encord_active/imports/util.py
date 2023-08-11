import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

import cv2
import pytz
import torch
import torchvision.ops
from torchvision.ops import masks_to_boxes

from encord_active.db.enums import AnnotationType

GMT_TIMEZONE = pytz.timezone("GMT")
DATETIME_STRING_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"


def get_timestamp():
    now = datetime.now()
    new_timezone_timestamp = now.astimezone(GMT_TIMEZONE)
    return new_timezone_timestamp.strftime(DATETIME_STRING_FORMAT)


def append_object_to_list(
    object_list: List[dict],
    annotation_type: AnnotationType,
    shape_data_list: List[Union[Dict[str, float], Dict[str, Dict[str, float]]]],
    ontology_object,
    confidence: Optional[float],
    object_hash: Optional[str],
) -> None:
    timestamp: str = get_timestamp()
    for shape_data_entry in shape_data_list:
        object_list.append(
            {
                "name": ontology_object.name,
                "color": ontology_object.color,
                "value": "_".join(ontology_object.name.lower().split()),
                "createdAt": timestamp,
                "createdBy": "robot@encord.com",
                "confidence": confidence or 1.0,
                "objectHash": object_hash or str(uuid.uuid4())[:8],
                "featureHash": ontology_object.feature_node_hash,
                "lastEditedAt": timestamp,
                "lastEditedBy": "robot@encord.com",
                "shape": str(annotation_type),
                "manualAnnotation": False,
                "reviews": [],
                str(annotation_type): shape_data_entry,
            }
        )


def coco_str_to_bitmask(rle: str, width: int, height: int) -> torch.Tensor:
    from pycocotools import mask
    bitmask = mask.decode({
        "counts": rle,
        "size": [width, height]
    })
    tensor = torch.from_numpy(bitmask)
    return tensor.T  # Convert to height, width format


def bitmask_to_bounding_box(bitmask: torch.Tensor) -> Dict[str, float]:
    x1, y1, x2, y2 = masks_to_boxes(bitmask.unsqueeze(0)).squeeze(0).tolist()
    height, width = bitmask.shape
    return {
        "x": x1 / width,
        "y": y1 / height,
        "w": (x2 + 1 - x1) / width,
        "h": (y2 + 1 - y1) / height,
    }


def bitmask_to_rotatable_bounding_box(bitmask: torch.Tensor) -> Dict[str, float]:
    raise ValueError


def bitmask_to_polygon(bitmask: torch.Tensor) -> List[Dict[str, Dict[str,float]]]:
    height, width = bitmask.shape
    npmask = bitmask.numpy()
    contours, _ = cv2.findContours(npmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    poly_dict_list = []
    for contour in contours:
        poly_dict = {}
        for idx, point in enumerate(contour):
            point = {
                "x": float(point[0][0]) / width,
                "y": float(point[0][1]) / height,
            }
            poly_dict[str(idx)] = point
        poly_dict_list.append(poly_dict)
    return poly_dict_list


def bitmask_to_encord_str(bitmask: torch.Tensor) -> str:
    from pycocotools import mask
    res = mask.encode(bitmask.T.numpy())
    return res["counts"].decode("utf-8")


def polygon_to_bitmask(
    polygon: Dict[str, Dict[str, float]],
    width: int,
    height: int
) -> torch.Tensor:
    raise ValueError
