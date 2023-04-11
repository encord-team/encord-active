import json
from pathlib import Path
from typing import Union
from urllib import parse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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


@app.get("/projects/{project}/label_rows/{lr_hash}/data_units/{du_hash}")
def read_item(project: str, lr_hash: str, du_hash: str, q: Union[str, None] = None):
    label_row_structure = ProjectFileStructure(target_path / project).label_row_structure(lr_hash)
    url = get_url(label_row_structure, du_hash)

    return {"url": url, **json.loads(label_row_structure.label_row_file.read_text())}


def get_url(label_row_structure: LabelRowStructure, du_hash: str):
    for data_unit in label_row_structure.iter_data_unit():
        if data_unit.hash == du_hash:
            return f"static/{parse.quote(data_unit.path.relative_to(label_row_structure.path.parents[2]).as_posix())}"
