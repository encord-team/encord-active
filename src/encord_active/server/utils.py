from functools import lru_cache, partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict
from urllib.parse import quote, unquote, urlparse

from shapely.affinity import rotate
from shapely.geometry import Polygon

from encord_active.lib.db.helpers.tags import to_grouped_tags
from encord_active.lib.embeddings.utils import SimilaritiesFinder
from encord_active.lib.labels.object import ObjectShape
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.metrics.utils import (
    MetricData,
    MetricScope,
    load_available_metrics,
)
from encord_active.lib.project.project_file_structure import (
    LabelRowStructure,
    ProjectFileStructure,
)
from encord_active.server.dependencies import ProjectFileStructureDep
from encord_active.server.settings import get_settings


class Metadata(TypedDict):
    annotator: Optional[str]
    labelClass: Optional[str]
    metrics: Dict[str, str]


@lru_cache
def load_project_metrics(project: ProjectFileStructure, scope: Optional[MetricScope] = None) -> List[MetricData]:
    return load_available_metrics(project.metrics, scope)


@lru_cache
def get_similarity_finder(embedding_type: EmbeddingType, project: ProjectFileStructureDep):
    return SimilaritiesFinder(embedding_type, project)


def _get_url(label_row_structure: LabelRowStructure, du_hash: str, frame: str) -> Optional[Tuple[str, Optional[float]]]:
    data_opt = next(label_row_structure.iter_data_unit(du_hash, int(frame)), None) or next(
        label_row_structure.iter_data_unit(du_hash, None), None
    )
    if data_opt:
        timestamp = None
        if data_opt.data_type == "video":
            timestamp = (float(int(frame)) + 0.5) / data_opt.frames_per_second
        signed_url = data_opt.signed_url
        if signed_url.startswith("file://"):
            image_path = Path(unquote(urlparse(signed_url).path)).absolute()
            settings = get_settings()
            root_path = label_row_structure.label_row_file_deprecated_for_migration().parents[3]
            try:
                relative_path = image_path.relative_to(root_path)
                signed_url = f"{settings.API_URL}/ea-static/{quote(relative_path.as_posix())}"
            except ValueError as ex:
                # Use hacky fallback
                approx_relative_path = image_path.as_posix().removeprefix(root_path.as_posix())
                if approx_relative_path.startswith("/"):
                    approx_relative_path = approx_relative_path[1:]
                signed_url = f"{settings.API_URL}/ea-static/{quote(approx_relative_path)}"

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
) -> dict:
    edit_url = row.pop("url", None)
    tags = row.pop("tags")
    identifier = row.pop("identifier")
    metadata = Metadata(
        labelClass=row.pop("object_class", None),
        annotator=row.pop("annotator", None),
        metrics=row,
    )

    label_row_structure = project_file_structure.label_row_structure(lr_hash)
    url = _get_url(label_row_structure, du_hash, frame)

    label_row = label_row_structure.label_row_json
    du = label_row["data_units"][du_hash]
    img_w, img_h = du["width"], du["height"]
    data_title = du.get("data_title", label_row.get("data_title"))

    labels: Dict[str, List[Dict]] = {"objects": [], "classifications": []}
    if label_row["data_type"] in {"video", "dicom"}:
        labels = du.get("labels", {}).get(str(int(frame)), labels)
    else:
        labels = du.get("labels", labels)

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

    return {
        "id": identifier,
        "url": url[0] if url is not None else None,
        "videoTimestamp": url[1] if url is not None else None,
        "dataTitle": data_title,
        "editUrl": edit_url,
        "metadata": metadata,
        "tags": to_grouped_tags(tags),
        "labels": labels,
    }
