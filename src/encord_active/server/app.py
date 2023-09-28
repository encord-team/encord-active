import logging
import webbrowser
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.engine import Engine
from starlette.responses import FileResponse, HTMLResponse

from encord_active.server.dependencies import (
    dep_engine,
    dep_engine_readonly,
    dep_settings,
    verify_premium,
    verify_token,
)

from ..imports.sandbox.sandbox_projects import IMAGES_PATH
from .routers import project2
from .settings import Env, Settings


def get_app(engine: Engine, settings: Settings) -> FastAPI:
    app = FastAPI()

    app.include_router(project2.router, dependencies=[Depends(verify_token)], prefix="/api")

    # Create read-only engine
    if engine.dialect.name == "postgresql":
        readonly_engine = engine.execution_options(
            isolation_level="AUTOCOMMIT",
            postgresql_readonly=True,
        )
    else:
        readonly_engine = engine.execution_options()

    # Hook dependencies
    app.dependency_overrides = {
        dep_engine: lambda: engine,
        dep_engine_readonly: lambda: readonly_engine,
        dep_settings: lambda: settings,
    }

    origins = [
        settings.ALLOWED_ORIGIN,
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5173/",
    ]

    is_dev = settings.ENV == Env.DEVELOPMENT

    if is_dev:
        logger = logging.getLogger(__name__)
        logger.info("Make sure you run the frontend separately.")
    else:
        frontend_build_path = Path(__file__).parent.parent / "frontend" / "dist" / settings.ENV.value

        if not frontend_build_path.exists() or not (frontend_build_path / "assets").exists():
            logger = logging.getLogger(__name__)
            logger.error("Cannot find frontend-components:")
            logger.error(f" {frontend_build_path} does not exist...")
            raise RuntimeError("Bad encord-active install, frontend-components are missing!!")

        app.mount(
            "/assets", StaticFiles(directory=frontend_build_path / "assets", follow_symlink=False), name="fe-assets"
        )

        @app.get("/{any:path}")
        @app.get("/index.html")
        def _index() -> FileResponse:
            return FileResponse(frontend_build_path / "index.html")

        @app.get("/favicon.ico")
        def _favicon_ico() -> FileResponse:
            return FileResponse(frontend_build_path / "favicon.ico", media_type="image/x-icon")

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
            webbrowser.open(settings.API_URL, new=0, autoraise=True)

    @app.get("/premium_available")
    async def premium_available() -> bool:
        try:
            await verify_premium(settings)
            return True
        except:
            return False

    @app.get("/")
    def health_check() -> bool:
        return True

    return app
