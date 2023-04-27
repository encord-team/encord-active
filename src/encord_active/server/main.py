from os import environ
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from natsort import natsorted
from pydantic import BaseModel

from encord_active.cli.utils.decorators import is_project
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.helpers.tags import (
    GroupedTags,
    all_tags,
    from_grouped_tags,
    to_grouped_tags,
)
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.embeddings.dimensionality_reduction import get_2d_embedding_data
from encord_active.lib.metrics.utils import MetricScope, filter_none_empty_metrics
from encord_active.lib.project.project_file_structure import ProjectFileStructure
from encord_active.server.utils import (
    get_metric_embedding_type,
    get_similarity_finder,
    load_project_metrics,
    to_item,
)

path = Path(environ["SERVER_START_PATH"])
if is_project(path):
    path = path.parent

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

app.mount("/static", StaticFiles(directory=path), name="static")


async def get_project_file_structure(project: str) -> ProjectFileStructure:
    return ProjectFileStructure(path / project)


ProjectFileStructureDep = Annotated[ProjectFileStructure, Depends(get_project_file_structure)]


@app.get("/projects/{project}/items_id_by_metric")
def read_item_ids(project: ProjectFileStructureDep, sort_by_metric: str, ascending: bool = True):
    DBConnection.set_project_file_structure(project)
    merged_metrics = MergedMetrics().all(marshall=False)

    column = [col for col in merged_metrics.columns if col.lower() == sort_by_metric.lower()][0]
    res = merged_metrics[[column]].dropna().sort_values(by=[column], ascending=ascending)

    return res.reset_index().rename({"identifier": "id", column: "value"}, axis=1).to_dict("records")


@app.get("/projects/{project}/tagged_items")
def tagged_items(project: ProjectFileStructureDep):
    DBConnection.set_project_file_structure(project)
    df = MergedMetrics().all(columns=["tags"]).reset_index()
    records = df[df["tags"].str.len() > 0].to_dict("records")
    return {record["identifier"]: to_grouped_tags(record["tags"]) for record in records}


@app.get("/projects/{project}/items/{id:path}")
def read_item(project: ProjectFileStructureDep, id: str):
    lr_hash, du_hash, frame, *_ = id.split("_")
    DBConnection.set_project_file_structure(project)
    row = MergedMetrics().get_row(id).dropna(axis=1).to_dict("records")[0]

    return to_item(row, project, lr_hash, du_hash, frame)


class ItemTags(BaseModel):
    id: str
    grouped_tags: GroupedTags


@app.put("/projects/{project}/item_tags")
def tag_items(project: ProjectFileStructureDep, payload: List[ItemTags]):
    DBConnection.set_project_file_structure(project)
    for item in payload:
        MergedMetrics().update_tags(item.id, from_grouped_tags(item.grouped_tags))


@app.get("/projects/{project}/similarities/{id}")
def get_similar_items(project: ProjectFileStructureDep, id: str, current_metric: str, page_size: Optional[int] = None):
    embedding_type = get_metric_embedding_type(project, current_metric)
    finder = get_similarity_finder(embedding_type, project.embeddings, page_size)
    nearest_images = finder.get_similarities(id)

    return [item["key"] for item in nearest_images]


@app.get("/projects/{project}/metrics")
def get_available_metrics(project: ProjectFileStructureDep, scope: Optional[MetricScope] = None):
    metrics = load_project_metrics(project, scope)
    non_empty_metrics = list(map(lambda i: i.name, filter(filter_none_empty_metrics, metrics)))
    return natsorted(non_empty_metrics)


@app.get("/projects/{project}/2d_embeddings/{current_metric}")
def get_2d_embeddings(project: ProjectFileStructureDep, current_metric: str):
    embedding_type = get_metric_embedding_type(project, current_metric)
    embeddings_df = get_2d_embedding_data(project.embeddings, embedding_type)

    if embeddings_df is None:
        raise HTTPException(
            status_code=404, detail=f"Embeddings of type: {embedding_type} were not found for project: {project}"
        )

    return embeddings_df.rename({"identifier": "id"}, axis=1).to_dict("records")


@app.get("/projects/{project}/tags")
def get_tags(project: ProjectFileStructureDep):
    DBConnection.set_project_file_structure(project)
    return to_grouped_tags(all_tags())
