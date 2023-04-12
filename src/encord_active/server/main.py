import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, TypedDict, Union
from urllib import parse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.helpers.tags import to_grouped_tags
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.embeddings.utils import SimilaritiesFinder
from encord_active.lib.metrics.metric import EmbeddingType
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


@app.get("/projects/{project}/label_rows/{lr_hash}/data_units/{du_hash}")
def read_item(project: str, lr_hash: str, du_hash: str, full_id: str):
    project_file_structure = ProjectFileStructure(target_path / project)

    DBConnection.set_project_path(project_file_structure.project_dir)
    row = MergedMetrics().get_row(full_id).dropna(axis=1).to_dict("records")[0]

    editUrl = row.pop("url")
    tags = row.pop("tags")
    metadata = Metadata(
        labelClass=row.pop("object_class", None),
        annotator=row.pop("annotator", None),
        metrics=row,
    )

    label_row_structure = project_file_structure.label_row_structure(lr_hash)
    url = _get_url(label_row_structure, du_hash)
    _extra = json.loads(label_row_structure.label_row_file.read_text())

    return {"url": url, "editUrl": editUrl, "metadata": metadata, "tags": to_grouped_tags(tags)}


@app.get("/projects/{project}/similarities/{id}")
def get_similar_items(project: str, id: str, embedding_type: EmbeddingType, page_size: int = 8):
    project_file_structure = ProjectFileStructure(target_path / project)
    finder = _get_similarity_finder(embedding_type, project_file_structure.embeddings, page_size)
    nearest_images = finder.get_similarities(id)
    print(nearest_images)

    return {}


@lru_cache
def _get_similarity_finder(embedding_type: EmbeddingType, path: Path, num_of_neighbors: int = 8):
    return SimilaritiesFinder(embedding_type, path, num_of_neighbors)
