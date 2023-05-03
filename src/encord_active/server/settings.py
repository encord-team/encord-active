from enum import Enum

from pydantic import BaseSettings


class Env(str, Enum):
    LOCAL = "local"
    PROD = "prod"


class Settings(BaseSettings):
    ENV: Env = Env.LOCAL

    class Config:
        env_file = ".env"


settings = Settings()
