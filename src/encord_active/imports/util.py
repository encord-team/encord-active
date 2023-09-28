import mimetypes
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import pytz
import torch
from encord.objects import Object
from pycocotools import mask
from torchvision.ops import masks_to_boxes

from encord_active.db.enums import AnnotationType

GMT_TIMEZONE = pytz.timezone("GMT")
DATETIME_STRING_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"


def get_mimetype(path: Path, fallback: str = "unknown") -> str:
    guess = mimetypes.guess_type(path)[0]
    if guess:
        return guess
    return f"{fallback}/{path.suffix[1:]}"


def get_timestamp():
    now = datetime.now()
    new_timezone_timestamp = now.astimezone(GMT_TIMEZONE)
    return new_timezone_timestamp.strftime(DATETIME_STRING_FORMAT)


_name_key_overrides: Dict[AnnotationType, str] = {
    AnnotationType.BOUNDING_BOX: "boundingBox",
    AnnotationType.ROTATABLE_BOUNDING_BOX: "rotatableBoundingBox",
}


def append_object_to_list(
    object_list: List[dict],
    annotation_type: AnnotationType,
    shape_data_list: Sequence[Union[Dict[str, float], Dict[str, Dict[str, float]], str]],
    ontology_object: Object,
    confidence: Optional[float],
    object_hash: Optional[str],
    email: str = "coco-import@encord.com",
) -> None:
    timestamp: str = get_timestamp()
    if AnnotationType(ontology_object.shape.value.lower()) != annotation_type:
        raise ValueError(
            f"Inconsistent generated encord ontology: {AnnotationType(ontology_object.shape)} != {annotation_type}"
        )
    for shape_data_entry in shape_data_list:
        object_list.append(
            {
                "name": ontology_object.name,
                "color": ontology_object.color,
                "value": "_".join(ontology_object.name.lower().split()),
                "createdAt": timestamp,
                "createdBy": email,
                "confidence": confidence or 1.0,
                "objectHash": object_hash or str(uuid.uuid4())[:8],
                "featureHash": ontology_object.feature_node_hash,
                "lastEditedAt": timestamp,
                "lastEditedBy": email,
                "shape": str(annotation_type),
                "manualAnnotation": False,
                "reviews": [],
                str(_name_key_overrides.get(annotation_type, annotation_type)): shape_data_entry,
            }
        )


def coco_str_to_bitmask(rle: str, width: int, height: int) -> torch.Tensor:
    bitmask = mask.decode({"counts": rle, "size": [width, height]})
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
    print("WARNING: Bitmask to rot bb conversion is not currently supported - useing bb conversion")
    bb = bitmask_to_bounding_box(bitmask)
    bb["theta"] = 0
    return bb


def bitmask_to_polygon(bitmask: torch.Tensor) -> List[Dict[str, Dict[str, float]]]:
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


def bitmask_to_encord_dict(bitmask: torch.Tensor) -> dict:
    res = mask.encode(bitmask.T.numpy())

    return {
        "rleString": res["counts"].decode("utf-8"),
        "width": res["size"][0],
        "height": res["size"][1],
        "top": 0,
        "left": 0,
    }


def polygon_to_bitmask(polygon: Dict[str, Dict[str, float]], width: int, height: int) -> torch.Tensor:
    raw_mask = np.zeros((width, height), dtype=np.uint8)
    raw_points = np.array(
        [[float(polygon[str(i)]["x"]) * width, float(polygon[str(i)]["y"]) * height] for i in range(len(polygon))]
    ).astype(np.int32)
    cv2.fillPoly(raw_mask, [raw_points], 1)
    return torch.tensor(raw_mask).bool()
