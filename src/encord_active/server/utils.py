import json
import uuid
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

from cachetools import LRUCache, cached
from shapely.affinity import rotate
from shapely.geometry import Polygon

from encord_active.lib.common.data_utils import url_to_file_path
from encord_active.lib.common.filtering import Filters, apply_filters
from encord_active.lib.common.utils import mask_to_polygon, rle_to_binary_mask
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.helpers.tags import GroupedTags
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.embeddings.utils import SimilaritiesFinder
from encord_active.lib.labels.object import ObjectShape
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.metrics.utils import (
    MetricData,
    MetricScope,
    load_available_metrics,
)
from encord_active.lib.model_predictions.reader import read_class_idx as _read_class_idx
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.project.project_file_structure import (
    LabelRowStructure,
    ProjectFileStructure,
)
from encord_active.server.dependencies import ProjectFileStructureDep


class Metadata(TypedDict):
    annotator: Optional[str]
    labelClass: Optional[str]
    metrics: Dict[str, str]


@cached(cache=LRUCache(maxsize=100))
def load_project_metrics(project: ProjectFileStructure, scope: Optional[MetricScope] = None) -> List[MetricData]:
    if scope == MetricScope.PREDICTION:
        return load_available_metrics(project.predictions / "object" / "metrics")
    return load_available_metrics(project.metrics, scope)


@cached(cache=LRUCache(maxsize=100))
def get_similarity_finder(embedding_type: EmbeddingType, project: ProjectFileStructureDep):
    return SimilaritiesFinder(embedding_type, project)


@cached(cache=LRUCache(maxsize=5))
def filtered_merged_metrics(project: ProjectFileStructure, filters: Filters, scope: Optional[MetricScope] = None):
    with DBConnection(project) as conn:
        merged_metrics = MergedMetrics(conn).all()

    return apply_filters(merged_metrics, filters, project, scope)


@cached(cache=LRUCache(maxsize=100))
def read_class_idx(predictions_dir: Path):
    return _read_class_idx(predictions_dir)


def _get_url(
    project_hash: uuid.UUID, label_row_structure: LabelRowStructure, du_hash: str, frame: str
) -> Optional[Tuple[str, Optional[float]]]:
    data_opt = next(label_row_structure.iter_data_unit(du_hash, int(frame)), None) or next(
        label_row_structure.iter_data_unit(du_hash, None), None
    )
    if data_opt:
        timestamp = None
        if data_opt.data_type == "video":
            timestamp = (float(int(frame)) + 0.5) / data_opt.frames_per_second
        signed_url = data_opt.signed_url
        file_path = url_to_file_path(signed_url, label_row_structure.project.project_dir)
        if file_path is not None:
            url_frame = 0 if data_opt.data_type == "video" else frame
            signed_url = f"/projects/{project_hash}/local-fs/{label_row_structure.label_hash}/{du_hash}/{url_frame}"

        return signed_url, timestamp
    return None


def _transform_object(object_: dict, img_w: int, img_h: int) -> Optional[dict]:
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
                [b["x"] * img_w, b["y"] * img_h],
                [(b["x"] + b["w"]) * img_w, b["y"] * img_h],
                [(b["x"] + b["w"]) * img_w, (b["y"] + b["h"]) * img_h],
                [b["x"] * img_w, (b["y"] + b["h"]) * img_h],
            ]
            rotated_polygon = rotate(Polygon(no_rotate_points), b["theta"])
            if rotated_polygon.exterior:
                points = {
                    i: {"x": c[0] / img_w, "y": c[1] / img_h} for i, c in enumerate(rotated_polygon.exterior.coords)
                }
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
    object_hash: Optional[str] = None,
) -> dict:
    # Derive encord url
    edit_url: Optional[str] = None
    project_meta = project_file_structure.load_project_meta()
    if project_meta.get("has_remote", False):
        project_hash = project_meta["project_hash"]
        edit_url = f"https://app.encord.com/label_editor/{du_hash}&{project_hash}/{frame}"

    tags = row.pop("tags", GroupedTags(data=[], label=[]))
    identifier = row.pop("identifier")
    metadata = Metadata(
        labelClass=row.pop("object_class", None),
        annotator=row.pop("annotator", None),
        metrics=row,
    )

    label_row_structure = project_file_structure.label_row_structure(lr_hash)
    url = _get_url(uuid.UUID(project_meta["project_hash"]), label_row_structure, du_hash, frame)

    label_row = label_row_structure.label_row_json
    du = label_row["data_units"][du_hash]
    img_w, img_h = du["width"], du["height"]
    data_title = du.get("data_title", label_row.get("data_title"))

    if label_row["data_type"] in {"video", "dicom"}:
        labels = du.get("labels", {}).get(str(int(frame)), None)
    else:
        labels = du.get("labels", None)

    if labels is not None:
        labels["objects"] = list(
            filter(None, map(partial(_transform_object, img_w=img_w, img_h=img_h), labels.get("objects", [])))
        )

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

    object_predictions = []
    classification_predictions = []
    if "is_true_positive" in row:
        if "iou" in row:
            object_predictions = build_item_object_predictions(
                row, project_file_structure, metadata, img_w, img_h, object_hash
            )
        else:
            classification_predictions = build_item_classification_predictions(row, project_file_structure, metadata)

    return {
        "id": identifier,
        "url": url[0] if url is not None else None,
        "videoTimestamp": url[1] if url is not None else None,
        "dataTitle": data_title,
        "editUrl": edit_url,
        "metadata": metadata,
        "tags": tags,
        "labels": labels or {"objects": [], "classifications": []},
        "predictions": {"objects": object_predictions, "classifications": classification_predictions},
    }


def build_item_object_predictions(
    row: Dict,
    project_file_structure: ProjectFileStructure,
    metadata: Metadata,
    img_w: int,
    img_h: int,
    object_hash: Optional[str] = None,
):
    objects: List[Dict] = []
    rle = row.pop("rle", None)
    if not object_hash:
        return []

    class_id = row.pop("class_id", None)
    if class_id is None:
        return []

    prediction_points = None
    if rle:
        mask = rle_to_binary_mask(json.loads(rle.replace("'", '"')))
        polygon, bbox = mask_to_polygon(mask)
        x, y, w, h = bbox
        if polygon is None:
            return []
        prediction_points = {i: {"x": x / img_w, "y": y / img_h} for i, (x, y) in enumerate(polygon)}
    else:
        x1, x2, y1, y2 = row["x1"], row["x2"], row["y1"], row["y2"]
        x, y = x1, y1
        w = x2 - x1
        h = y2 - y1

    x, y, w, h = (x / img_w, y / img_h, w / img_w, h / img_h)

    bbox_points = {
        0: {"x": x, "y": y},
        1: {"x": x + w, "y": y},
        2: {"x": x + w, "y": y + h},
        3: {"x": x, "y": y + h},
    }

    for key in [
        "Unnamed: 0",
        "img_id",
        "x1",
        "x2",
        "y1",
        "y2",
    ]:
        row.pop(key, None)

    is_true_positive = row["is_true_positive"]

    class_idx = read_class_idx(project_file_structure.predictions / MainPredictionType.OBJECT.value)
    class_name = row.get("class_name", None)

    metadata["annotator"] = "Prediction"
    metadata["labelClass"] = class_name

    objects.append(
        {
            "name": class_name,
            "color": "#22c55e" if is_true_positive else "#ef4444",
            "shape": "polygon",
            "confidence": row["confidence"],
            "objectHash": object_hash,
            "featureHash": class_idx[str(class_id)]["featureHash"],
            "points": prediction_points,
            "boundingBoxPoints": bbox_points,
        }
    )

    return objects


def build_item_classification_predictions(
    row: Dict,
    project_file_structure: ProjectFileStructure,
    metadata: Metadata,
):
    classifications: List[Dict] = []
    class_id = row.pop("class_id", None)

    if class_id is None:
        return classifications

    for key in ["Unnamed: 0", "false_positive_reason", "img_id", "gt_class_id"]:
        row.pop(key, None)

    row["Ground Truth Class"] = row.pop("gt_class_name", None)

    class_idx = read_class_idx(project_file_structure.predictions / MainPredictionType.CLASSIFICATION.value)
    class_name = row.pop("class_name", None)

    is_true_positive = row.pop("is_true_positive")
    metadata["annotator"] = "Correct Classifictaion" if is_true_positive else "Misclassification"
    metadata["labelClass"] = class_name

    classifications.append(
        {
            "name": class_name,
            "confidence": row["confidence"],
            "featureHash": class_idx[str(class_id)]["featureHash"],
        }
    )

    return classifications
