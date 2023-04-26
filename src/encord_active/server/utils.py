import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, TypedDict
from urllib import parse

from shapely.affinity import rotate
from shapely.geometry import Polygon

from encord_active.lib.db.helpers.tags import to_grouped_tags
from encord_active.lib.db.predictions import BoundingBox
from encord_active.lib.embeddings.utils import SimilaritiesFinder
from encord_active.lib.labels.object import ObjectShape
from encord_active.lib.metrics.metric import EmbeddingType
from encord_active.lib.metrics.utils import (
    MetricScope,
    get_embedding_type,
    load_available_metrics,
)
from encord_active.lib.project.project_file_structure import (
    LabelRowStructure,
    ProjectFileStructure,
)


class Metadata(TypedDict):
    annotator: Optional[str]
    labelClass: Optional[str]
    metrics: Dict[str, str]


@lru_cache
def load_project_metrics(project: ProjectFileStructure, scope: Optional[MetricScope] = None):
    return load_available_metrics(project.metrics, scope)


def get_metric_embedding_type(project: ProjectFileStructure, metric_name: str):
    metrics = load_project_metrics(project)
    metric_data = [metric for metric in metrics if metric.name.lower() == metric_name.lower()][0]
    return get_embedding_type(metric_data.meta.annotation_type)


@lru_cache
def get_similarity_finder(embedding_type: EmbeddingType, path: Path, num_of_neighbors: int = 8):
    return SimilaritiesFinder(embedding_type, path, num_of_neighbors)


def _get_url(label_row_structure: LabelRowStructure, du_hash: str):
    for data_unit in label_row_structure.iter_data_unit():
        if data_unit.hash == du_hash:
            return f"static/{parse.quote(data_unit.path.relative_to(label_row_structure.path.parents[2]).as_posix())}"


def _transform_object(object_: dict):
    if object_["shape"] == ObjectShape.POLYGON:
        p = object_.get("polygon", {})
        if not p:
            return None
    elif object_["shape"] == ObjectShape.BOUNDING_BOX:
        b = object_.get("boundingBox", {})
        if not b:
            return None
        object_["polygon"] = {
            0: {"x": b["x"], "y": b["y"]},
            1: {"x": b["x"] + b["w"], "y": b["y"]},
            2: {"x": b["x"] + b["w"], "y": b["y"] + b["h"]},
            3: {"x": b["x"], "y": b["y"] + b["h"]},
        }
        del object_["boundingBox"]
    elif object_["shape"] == ObjectShape.ROTATABLE_BOUNDING_BOX:
        b = object_["rotatableBoundingBox"]
        no_rotate_points = [
            [b["x"], b["y"]],
            [b["x"] + b["w"], b["y"]],
            [b["x"] + b["w"], b["y"] + b["h"]],
            [b["x"], b["y"] + b["h"]],
        ]
        rotated_polygon = rotate(Polygon(no_rotate_points), b["theta"])
        if not rotated_polygon or not rotated_polygon.exterior:
            return None
        object_["polygon"] = {i: {"x": c[0], "y": c[1]} for i, c in enumerate(rotated_polygon.exterior.coords)}
        del object_["rotatableBoundingBox"]
    else:
        return None
    return object_


def to_item(row: Dict, project_file_structure: ProjectFileStructure, lr_hash: str, du_hash: str, frame: str):
    editUrl = row.pop("url")
    tags = row.pop("tags")
    identifier = row.pop("identifier")
    metadata = Metadata(
        labelClass=row.pop("object_class", None),
        annotator=row.pop("annotator", None),
        metrics=row,
    )

    label_row_structure = project_file_structure.label_row_structure(lr_hash)
    url = _get_url(label_row_structure, du_hash)

    label_row = json.loads(label_row_structure.label_row_file.read_text())
    du = label_row["data_units"][du_hash]
    data_title = du.get("data_title", label_row.get("data_title"))

    if label_row["data_type"] in {"video", "dicom"}:
        labels = du.get("labels", {}).get(frame, {"objects": [], "classifications": []})
    else:
        labels = du.get("labels", {"objects": [], "classifications": []})

    labels["objects"] = list(filter(None, map(_transform_object, labels.get("objects", []))))

    try:
        classifications = labels.get("classifications")
        if classifications and metadata["labelClass"]:
            classification_hash = classifications[0]["classificationHash"]
            classification = label_row["classification_answers"][classification_hash]["classifications"][0]
            metadata["labelClass"] += f": {classification['answers'][0]['name']}"
    except:
        pass

    return {
        "id": identifier,
        "url": url,
        "data_title": data_title,
        "editUrl": editUrl,
        "metadata": metadata,
        "tags": to_grouped_tags(tags),
        "labels": labels,
    }
