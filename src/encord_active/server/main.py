import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, TypedDict
from urllib import parse

from encord_active_components.components.explorer import GroupedTags
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from natsort import natsorted
from pydantic import BaseModel

from encord_active.app.common.components.tags.utils import all_tags
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.helpers.tags import from_grouped_tags, to_grouped_tags
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.embeddings.dimensionality_reduction import get_2d_embedding_data
from encord_active.lib.embeddings.utils import SimilaritiesFinder
from encord_active.lib.metrics.metric import EmbeddingType
from encord_active.lib.metrics.utils import (
    MetricScope,
    filter_none_empty_metrics,
    get_embedding_type,
    load_available_metrics,
)
from encord_active.lib.project.project_file_structure import (
    LabelRowStructure,
    ProjectFileStructure,
)

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

target_path = Path("/Users/encord/work/datasets")

app.mount("/static", StaticFiles(directory=target_path), name="static")


def _get_url(label_row_structure: LabelRowStructure, du_hash: str):
    for data_unit in label_row_structure.iter_data_unit():
        if data_unit.hash == du_hash:
            return f"static/{parse.quote(data_unit.path.relative_to(label_row_structure.path.parents[2]).as_posix())}"


class Metadata(TypedDict):
    annotator: Optional[str]
    labelClass: Optional[str]
    metrics: Dict[str, str]


@app.get("/projects/{project}/items_id_by_metric")
def read_item_ids(project: str, sort_by_metric: str, ascending: bool = True):
    project_file_structure = ProjectFileStructure(target_path / project)
    DBConnection.set_project_path(project_file_structure.project_dir)
    merged_metrics = MergedMetrics().all(marshall=False)

    column = [col for col in merged_metrics.columns if col.lower() == sort_by_metric.lower()][0]
    res = merged_metrics[[column]].dropna().sort_values(by=[column], ascending=ascending)

    return res.reset_index().rename({"identifier": "id", column: "value"}, axis=1).to_dict("records")


@app.get("/projects/{project}/items/{id}")
def read_item(project: str, id: str):
    lr_hash, du_hash, frame, *obj_hash = id.split("_")
    project_file_structure = ProjectFileStructure(target_path / project)

    DBConnection.set_project_path(project_file_structure.project_dir)
    row = MergedMetrics().get_row(id).dropna(axis=1).to_dict("records")[0]

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


class ItemTags(BaseModel):
    id: str
    grouped_tags: GroupedTags


@app.put("/projects/{project}/item_tags")
def tag_items(project: str, item_tags: ItemTags):
    project_file_structure = ProjectFileStructure(target_path / project)
    DBConnection.set_project_path(project_file_structure.project_dir)
    MergedMetrics().update_tags(item_tags.id, from_grouped_tags(item_tags.grouped_tags))


@lru_cache
def _get_similarity_finder(embedding_type: EmbeddingType, path: Path, num_of_neighbors: int = 8):
    return SimilaritiesFinder(embedding_type, path, num_of_neighbors)


@app.get("/projects/{project}/similarities/{id}")
def get_similar_items(project: str, id: str, current_metric: str, page_size: Optional[int] = None):
    project_file_structure = ProjectFileStructure(target_path / project)
    embedding_type = _get_metric_embedding_type(project, current_metric)
    finder = _get_similarity_finder(embedding_type, project_file_structure.embeddings, page_size)
    nearest_images = finder.get_similarities(id)

    return [item["key"] for item in nearest_images]


@app.get("/projects/{project}/metrics")
def get_available_metrics(project: str, scope: Optional[MetricScope] = None):
    metrics = _load_project_metrics(project, scope)
    non_empty_metrics = list(map(lambda i: i.name, filter(filter_none_empty_metrics, metrics)))
    return natsorted(non_empty_metrics)


@app.get("/projects/{project}/2d_embeddings/{current_metric}")
def get_2d_embeddings(project: str, current_metric: str):
    project_file_structure = ProjectFileStructure(target_path / project)
    embedding_type = _get_metric_embedding_type(project, current_metric)
    embeddings_df = get_2d_embedding_data(project_file_structure.embeddings, embedding_type)

    if embeddings_df is None:
        raise ValueError(f"Embeddings of type: {embedding_type} were not found for project: {project}")

    return embeddings_df.rename({"identifier": "id"}, axis=1).to_dict("records")


@lru_cache
def _load_project_metrics(project_name: str, scope: Optional[MetricScope] = None):
    project_file_structure = ProjectFileStructure(target_path / project_name)
    return load_available_metrics(project_file_structure.metrics, scope)


def _get_metric_embedding_type(project_name: str, metric_name: str):
    metrics = _load_project_metrics(project_name)
    metric_data = [metric for metric in metrics if metric.name.lower() == metric_name.lower()][0]
    return get_embedding_type(metric_data.meta.annotation_type)


@app.get("/projects/{project}/tags")
def get_tags(project: str):
    project_file_structure = ProjectFileStructure(target_path / project)
    DBConnection.set_project_path(project_file_structure.project_dir)
    foo = all_tags()
    return to_grouped_tags(foo)
