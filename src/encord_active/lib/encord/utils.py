import uuid
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from json import dumps as json_dumps
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union

from encord import EncordUserClient
from encord.http.constants import RequestsSettings
from encord.objects.classification import Classification
from encord.objects.common import NestableOption, RadioAttribute
from encord.objects.ontology_object import Object
from encord.orm.label_row import LabelRowMetadata
from encord.orm.project import Project
from encord.project import ObjectShape

from encord_active.lib.common.time import get_timestamp
from encord_active.lib.db.predictions import Point
from encord_active.lib.labels.object import BoxShapes, ObjectShape

BBOX_KEYS = {"x", "y", "h", "w"}


def handle_enum_and_datetime(label_row_meta: LabelRowMetadata):
    def convert_value(obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    return dict((k, convert_value(v)) for k, v in asdict(label_row_meta).items())


class ProjectQuery(TypedDict, total=False):
    title_eq: str
    title_like: str
    desc_eq: str
    desc_like: str
    created_before: Union[str, datetime]
    created_after: Union[str, datetime]
    edited_before: Union[str, datetime]
    edited_after: Union[str, datetime]


def get_client(ssh_key_path: Path):
    return EncordUserClient.create_with_ssh_private_key(
        ssh_key_path.read_text(encoding="utf-8"),
        requests_settings=RequestsSettings(max_retries=5),
    )


def get_encord_project(ssh_key_path: Union[str, Path], project_hash: str):
    if isinstance(ssh_key_path, str):
        ssh_key_path = Path(ssh_key_path)
    client = get_client(ssh_key_path)
    return client.get_project(project_hash)


def get_encord_projects(ssh_key_path: Path, query: Optional[ProjectQuery] = None) -> List[Project]:
    client = get_client(ssh_key_path)
    if query is None:  # Get all projects
        query = ProjectQuery()
    projects: List[Project] = list(map(lambda x: x["project"], client.get_projects(**query)))
    return projects


def get_projects_json(ssh_key_path: Path, query: Optional[ProjectQuery] = None) -> str:
    projects = get_encord_projects(ssh_key_path, query)
    return json_dumps({p.project_hash: p.title for p in projects}, indent=2)


def lower_snake_case(s: str):
    return "_".join(s.lower().split())


def make_object_dict(
    ontology_object: Object,
    object_data: Union[Point, List[Point], Dict[str, float], None] = None,
    object_hash: Optional[str] = None,
) -> dict:
    """
    :type ontology_object: The ontology object to associate with the ``object_data``.
    :type object_data: The data to put in the object dictionary. This has to conform
        with the ``shape`` parameter defined in
        ``ontology_object["shape"]``.
        - ``shape == "point"``: For key-points, the object data should be
          a tuple with x, y coordinates as floats.
        - ``shape == "bounding_box"``: For bounding boxes, the
          ``object_data`` needs to be a dict with info:
          {"x": float, "y": float, "h": float, "w": float} specifying the
          top right corner of the box (x, y) and the height and width of
          the bounding box.
        - ``shape in ("polygon", "polyline")``: For polygons and
          polylines, the format is an iterable  of points:
          [(x, y), ...] specifying (ordered) points in the
          polygon/polyline.
          If ``object_hash`` is none, a new hash will be generated.
    :type object_hash: If you want the object to have the same id across frames (for
        videos only), you can specify the object hash, which need to be
        an eight-character hex string (e.g., use
        ``str(uuid.uuid4())[:8]`` or the ``objectHash`` from an
        associated object.
    :returns: An object dictionary conforming with the Encord label row data format.
    """
    if object_hash is None:
        object_hash = str(uuid.uuid4())[:8]

    timestamp: str = get_timestamp()
    shape = ontology_object.shape

    object_dict = {
        "name": ontology_object.name,
        "color": ontology_object.color,
        "value": lower_snake_case(ontology_object.name),
        "createdAt": timestamp,
        "createdBy": "robot@cord.tech",
        "confidence": 1,
        "objectHash": object_hash,
        "featureHash": ontology_object.feature_node_hash,
        "lastEditedAt": timestamp,
        "lastEditedBy": "robot@encord.com",
        "shape": shape.value,
        "manualAnnotation": False,
        "reviews": [],
    }

    if shape.value in [ObjectShape.POLYGON, ObjectShape.POLYLINE] and object_data:
        if not isinstance(object_data, list):
            raise ValueError(f"The `object_data` for {shape} should be a list of points.")

        object_dict[shape.value] = {
            str(i): {"x": round(x, 4), "y": round(y, 4)} for i, (x, y) in enumerate(object_data)
        }

    elif shape.value == ObjectShape.KEY_POINT:
        if not isinstance(object_data, tuple):
            raise ValueError(f"The `object_data` for {shape} should be a tuple.")
        if len(object_data) != 2:
            raise ValueError(f"The `object_data` for {shape} should have two coordinates.")
        if not isinstance(object_data[0], float):
            raise ValueError(f"The `object_data` for {shape} should contain floats.")

        object_dict[shape.value] = {"0": {"x": round(object_data[0], 4), "y": round(object_data[1], 4)}}

    elif shape.value in BoxShapes:
        if not isinstance(object_data, dict):
            raise ValueError(f"The `object_data` for {shape} should be a dictionary.")
        if len(BBOX_KEYS.intersection(object_data.keys())) != 4:
            raise ValueError(f"The `object_data` for {shape} should have keys {BBOX_KEYS}.")
        if not isinstance(object_data["x"], float):
            raise ValueError(f"The `object_data` for {shape} should float values.")

        box = {k: round(v, 4) for k, v in object_data.items()}
        if shape.value == ObjectShape.ROTATABLE_BOUNDING_BOX:
            if "theta" not in object_data:
                raise ValueError(f"The `object_data` for {shape} should contain a `theta` field.")

            object_dict["rotatableBoundingBox"] = {**box, "theta": object_data["theta"]}
        else:
            object_dict["boundingBox"] = box

    return object_dict


def make_classification_dict_and_answer_dict(
    classification: Classification,
    attribute: RadioAttribute,
    option: NestableOption,
    classification_hash: Optional[str] = None,
):
    """
    :type ontology_class: The ontology classification
    :type answers: A radio attribute "question"
    :type option: The answer to the question
    :type classification_hash: If a classification should persist with the same id over
                               multiple frames (for videos), you can reuse the
                               ``classificationHash`` of a classifications from a
                               previous frame.

    :returns: A classification and an answer dictionary conforming with the Encord label
              row data format.
    """
    if classification_hash is None:
        classification_hash = str(uuid.uuid4())[:8]

    answers = [
        {
            "featureHash": option.feature_node_hash,
            "name": option.label,
            "value": option.value,
        }
    ]

    classification_dict = {
        "classificationHash": classification_hash,
        "confidence": 1,
        "createdAt": get_timestamp(),
        "createdBy": "robot@encord.com",
        "featureHash": classification.feature_node_hash,
        "manualAnnotation": False,
        "name": attribute.name,
        "value": lower_snake_case(attribute.name),
    }

    classification_answer = {
        "classificationHash": classification_hash,
        "classifications": [
            {
                "answers": answers,
                "featureHash": attribute.feature_node_hash,
                "manualAnnotation": False,
                "name": attribute.name,
                "value": lower_snake_case(attribute.name),
            }
        ],
    }

    return classification_dict, classification_answer
