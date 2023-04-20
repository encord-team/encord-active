import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, TypedDict
from urllib import parse

from encord_active_components.components.explorer import GroupedTags

from encord_active.lib.db.helpers.tags import to_grouped_tags
from encord_active.lib.embeddings.utils import SimilaritiesFinder
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


def to_item(row: Dict, project_file_structure: ProjectFileStructure, lr_hash: str, du_hash: str):
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
    labels = label_row["data_units"][du_hash]["labels"]

    return {
        "id": identifier,
        "url": url,
        "editUrl": editUrl,
        "metadata": metadata,
        "tags": to_grouped_tags(tags),
        "labels": labels,
    }
