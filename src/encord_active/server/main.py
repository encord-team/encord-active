import logging
import webbrowser
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from encord_active.cli.utils.decorators import find_child_projects, is_project
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.model_predictions.reader import read_prediction_files
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.project.project_file_structure import ProjectFileStructure
from encord_active.lib.project.sandbox_projects.sandbox_projects import IMAGES_PATH
from encord_active.server.dependencies import verify_premium, verify_token
from encord_active.server.utils import get_similarity_finder

from .routers import project, project2
from .settings import Env, get_settings

app = FastAPI()

app.include_router(project.router, dependencies=[Depends(verify_token)])
app.include_router(project2.router, dependencies=[Depends(verify_token)])

origins = [get_settings().ALLOWED_ORIGIN, "http://localhost:3000", "http://localhost:5173", "http://localhost:5173/"]

is_dev = get_settings().ENV == Env.DEVELOPMENT

if is_dev:
    logger = logging.getLogger(__name__)
    logger.info("Make sure you run the frontend separately.")
else:
    import encord_active_components as frontend_components

    frontend_build_path = Path(frontend_components.__file__).parent / "frontend" / "dist" / get_settings().ENV.value

    if not frontend_build_path.exists() or not (frontend_build_path / "assets").exists():
        logger = logging.getLogger(__name__)
        logger.error("Cannot find frontend-components:")
        logger.error(f" {frontend_build_path} does not exist...")
        raise RuntimeError("Bad encord-active install, frontend-components are missing!!")

    app.mount("/assets", StaticFiles(directory=frontend_build_path / "assets", follow_symlink=False), name="fe-assets")

    @app.get("/")
    @app.get("/index.html")
    def _index():
        return FileResponse(frontend_build_path / "index.html")

    @app.get("/favicon.ico")
    def _favicon_ico():
        return FileResponse(frontend_build_path / "favicon.ico")


app.mount("/ea-sandbox-static", StaticFiles(directory=IMAGES_PATH, follow_symlink=False), name="sandbox-static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    root_path = get_settings().SERVER_START_PATH
    paths = [root_path] if is_project(root_path) else find_child_projects(root_path)

    for path in paths:
        project_file_structure = ProjectFileStructure(path)
        for prediction_type in MainPredictionType:
            try:
                read_prediction_files(project_file_structure, prediction_type)
            except:
                pass
        for embedding_type in EmbeddingType:
            try:
                get_similarity_finder(embedding_type, project_file_structure)
            except:
                pass

    if not is_dev:
        webbrowser.open(get_settings().API_URL, new=0, autoraise=True)


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
