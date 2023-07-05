from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from encord_active.cli.utils.decorators import find_child_projects, is_project
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.model_predictions.reader import read_prediction_files
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.project.project_file_structure import ProjectFileStructure
from encord_active.server.dependencies import verify_premium
from encord_active.server.utils import get_similarity_finder

from .routers import project
from .settings import get_settings

app = FastAPI()

app.include_router(project.router)

origins = ["http://localhost:5173", "http://localhost:8501", get_settings().ALLOWED_ORIGIN]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ea-static", StaticFiles(directory=get_settings().SERVER_START_PATH, follow_symlink=True), name="static")


@app.on_event("startup")
async def on_startup():
    root_path = get_settings().SERVER_START_PATH
    paths = [root_path] if is_project(root_path) else find_child_projects(root_path)

    for path in paths:
        project_file_structure = ProjectFileStructure(path)
        for prediction_type in MainPredictionType:
            read_prediction_files(project_file_structure, prediction_type)
        for embedding_type in EmbeddingType:
            try:
                get_similarity_finder(embedding_type, project_file_structure)
            except:
                pass


@app.get("/premium_available")
async def premium_available():
    try:
        await verify_premium()
        return True
    except:
        return False


@app.get("/")
def health_check() -> bool:
    return True
