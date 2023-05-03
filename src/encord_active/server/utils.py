import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, TypedDict
from urllib import parse

from shapely.affinity import rotate
from shapely.geometry import Polygon

from encord_active.lib.db.helpers.tags import to_grouped_tags
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


def _get_url(label_row_structure: LabelRowStructure, du_hash: str, frame: str):
    data_unit = next(label_row_structure.iter_data_unit(du_hash, int(frame)), None) or next(
        label_row_structure.iter_data_unit(du_hash), None
    )
    if data_unit:
        return f"static/{parse.quote(data_unit.path.relative_to(label_row_structure.path.parents[2]).as_posix())}"


def _transform_object(object_: dict):
    shape = object_["shape"]
    points = None

    try:
        if shape == ObjectShape.POLYGON:
            points = object_.pop("polygon")
        elif shape == ObjectShape.POLYLINE:
            points = object_.pop("polyline")
        elif shape == ObjectShape.BOUNDING_BOX:
            b = object_.pop("boundingBox")
            points = {
                0: {"x": b["x"], "y": b["y"]},
                1: {"x": b["x"] + b["w"], "y": b["y"]},
                2: {"x": b["x"] + b["w"], "y": b["y"] + b["h"]},
                3: {"x": b["x"], "y": b["y"] + b["h"]},
            }
        elif shape == ObjectShape.ROTATABLE_BOUNDING_BOX:
            b = object_.pop("rotatableBoundingBox")
            no_rotate_points = [
                [b["x"], b["y"]],
                [b["x"] + b["w"], b["y"]],
                [b["x"] + b["w"], b["y"] + b["h"]],
                [b["x"], b["y"] + b["h"]],
            ]
            rotated_polygon = rotate(Polygon(no_rotate_points), b["theta"])
            if rotated_polygon.exterior:
                points = {i: {"x": c[0], "y": c[1]} for i, c in enumerate(rotated_polygon.exterior.coords)}
        elif shape == ObjectShape.KEY_POINT:
            points = object_.pop("point")

        if not points:
            raise Exception

        return {**object_, "shape": shape, "points": points}
    except:
        return None


def to_item(
    row: Dict,
    project_file_structure: ProjectFileStructure,
    lr_hash: str,
    du_hash: str,
    frame: str,
):
    editUrl = row.pop("url")
    tags = row.pop("tags")
    identifier = row.pop("identifier")
    metadata = Metadata(
        labelClass=row.pop("object_class", None),
        annotator=row.pop("annotator", None),
        metrics=row,
    )

    label_row_structure = project_file_structure.label_row_structure(lr_hash)
    url = _get_url(label_row_structure, du_hash, frame)

    label_row = json.loads(label_row_structure.label_row_file.read_text())
    du = label_row["data_units"][du_hash]
    data_title = du.get("data_title", label_row.get("data_title"))

    labels: Dict[str, List[Dict]] = {"objects": [], "classifications": []}
    if label_row["data_type"] in {"video", "dicom"}:
        labels = du.get("labels", {}).get(str(int(frame)), labels)
    else:
        labels = du.get("labels", labels)

    labels["objects"] = list(filter(None, map(_transform_object, labels.get("objects", []))))

    try:
        classifications = labels.get("classifications")
        if classifications and metadata["labelClass"]:
            clf_instance = classifications[0]
            question = clf_instance["name"]
            clf_hash = clf_instance["classificationHash"]
            classification = label_row["classification_answers"][clf_hash]["classifications"][0]
            metadata["labelClass"] = f"{question}: {classification['answers'][0]['name']}"
    except:
        pass

    return {
        "id": identifier,
        "url": url,
        "dataTitle": data_title,
        "editUrl": editUrl,
        "metadata": metadata,
        "tags": to_grouped_tags(tags),
        "labels": labels,
    }
