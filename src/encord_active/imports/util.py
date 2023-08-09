import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

import pytz
import torch

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


def coco_str_to_bitmask(coco: str, width: int, height: int) -> torch.Tensor:
    raise ValueError


def bitmask_to_bounding_box(bitmask: torch.Tensor) -> Dict[str, float]:
    raise ValueError


def bitmask_to_rotatable_bounding_box(bitmask: torch.Tensor) -> Dict[str, float]:
    raise ValueError


def bitmask_to_polygon(bitmask: torch.Tensor) -> Dict[str, float]:
    raise ValueError


def bitmask_to_encord_str(bitmask: torch.Tensor) -> str:
    raise ValueError


def polygon_to_bitmask(polygon: Dict[str, Dict[str, float]]) -> torch.Tensor:
    raise ValueError
