import logging
import webbrowser
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from sqlalchemy.engine import Engine
from starlette.responses import FileResponse

from encord_active.lib.project.sandbox_projects.sandbox_projects import IMAGES_PATH
from encord_active.server.dependencies import (
    dep_engine,
    dep_oauth2_scheme,
    verify_premium,
    verify_token,
)

from .routers import project2
from .settings import Env, get_settings


def get_app(engine: Engine, oauth2_scheme: OAuth2PasswordBearer) -> FastAPI:
    app = FastAPI()

    # app.include_router(project.router, dependencies=[Depends(verify_token)])
    app.include_router(project2.router, dependencies=[Depends(verify_token)])

    # Hook dependencies
    app.dependency_overrides = {
        dep_engine: lambda: engine,
        dep_oauth2_scheme: lambda: oauth2_scheme,
    }

    origins = [
        get_settings().ALLOWED_ORIGIN,
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5173/",
    ]

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

        app.mount(
            "/assets", StaticFiles(directory=frontend_build_path / "assets", follow_symlink=False), name="fe-assets"
        )

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
    async def on_startup() -> None:
        if not is_dev:
            webbrowser.open(get_settings().API_URL, new=0, autoraise=True)

    @app.get("/premium_available")
    async def premium_available() -> bool:
        try:
            await verify_premium()
            return True
        except:
            return False

    @app.get("/")
    def health_check() -> bool:
        return True

    return app
