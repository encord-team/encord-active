from enum import Enum
from os import environ
from pathlib import Path
from typing import Optional

from cachetools import LRUCache, cached
from pydantic import BaseSettings

from encord_active.cli.utils.decorators import is_project


class AvailableSandboxProjects(str, Enum):
    ALL = "all"
    BASE = "base"
    NONE = "none"


class Env(str, Enum):
    DEVELOPMENT = "development"
    PACKAGED = "packaged"
    SANDBOX = "sandbox"
    PRODUCTION = "production"


class Settings(BaseSettings):
    ENV: Env = Env.PACKAGED
    DEPLOYMENT_NAME: Optional[str] = None
    API_URL: str = "http://localhost:8000"
    ALLOWED_ORIGIN: Optional[str] = None
    JWT_SECRET: Optional[str] = None
    SERVER_START_PATH: Path
    AVAILABLE_SANDBOX_PROJECTS: AvailableSandboxProjects = AvailableSandboxProjects.ALL

    class Config:
        env_file = ".env"


@cached(cache=LRUCache(maxsize=10))
def get_settings():
    path = Path(environ.get("SERVER_START_PATH", "/data"))
    return Settings(SERVER_START_PATH=path.parent if is_project(path) else path)
