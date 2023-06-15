from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from encord_active.server.dependencies import verify_premium

from .routers import project
from .settings import get_settings, Env

app = FastAPI()

app.include_router(project.router)

origins = ["http://localhost:5173", "http://localhost:8501", get_settings().ALLOWED_ORIGIN]

if get_settings().ENV != Env.LOCAL:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/ea-static", StaticFiles(directory=get_settings().SERVER_START_PATH, follow_symlink=True), name="static")


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
