from enum import Enum
from typing import Optional

from pydantic import BaseSettings


class Env(str, Enum):
    LOCAL = "local"
    PROD = "prod"


class Settings(BaseSettings):
    ENV: Env = Env.LOCAL
    DEPLOYMENT_NAME: Optional[str] = None
    API_URL: str = "http://localhost:8000"
    ALLOWED_ORIGIN: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()
