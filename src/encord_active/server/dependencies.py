import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, Form, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import decode
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlalchemy.future.engine import OptionEngine

from ..cli.app_config import app_config
from .settings import Env, Settings


class DataItem(BaseModel):
    du_hash: uuid.UUID
    frame: int

    def pack(self) -> str:
        return f"{self.du_hash}_{self.frame}"


class DataOrAnnotateItem(BaseModel):
    du_hash: uuid.UUID
    frame: int
    annotation_hash: Optional[str]

    def pack(self) -> str:
        if self.annotation_hash is None:
            return f"{self.du_hash}_{self.frame}"
        return f"{self.du_hash}_{self.frame}_{self.annotation_hash}"


def parse_data_item(data_item: str) -> DataItem:
    segments = data_item.split("_")
    if len(segments) != 2:
        raise ValueError(f"DataItem expects 2 segments: {data_item}")
    du_hash, frame = segments
    return DataItem(du_hash=uuid.UUID(du_hash), frame=int(frame))


def parse_optional_data_or_annotate_item(item: Annotated[Optional[str], Form()] = None) -> Optional[DataOrAnnotateItem]:
    return None if item is None else parse_data_or_annotate_item(item)


def parse_data_or_annotate_item(item: str) -> DataOrAnnotateItem:
    segments = item.split("_")
    if len(segments) < 2:
        raise ValueError(f"Item expects at least 2 segments: {item}")
    du_hash, frame, *annotation_hash_segments = segments
    annotation_hash = None if len(annotation_hash_segments) == 0 else "_".join(annotation_hash_segments)
    if annotation_hash is not None and len(annotation_hash) != 8:
        raise ValueError(f"Item annotation_hash should be exactly 8 characters: {annotation_hash}")
    return DataOrAnnotateItem(du_hash=uuid.UUID(du_hash), frame=int(frame), annotation_hash=annotation_hash)


def dep_engine() -> Engine:
    raise RuntimeError("Missing Engine")


def dep_engine_readonly() -> OptionEngine:
    raise RuntimeError("Missing ReadOnly Engine")


def dep_settings() -> Settings:
    raise RuntimeError("Missing Settings")


def dep_ssh_key(settings: Annotated[Settings, Depends(dep_settings)]) -> str:
    if settings.SSH_KEY:
        return settings.SSH_KEY
    ssh_key_path = app_config.get_ssh_key()
    if ssh_key_path is None:
        raise RuntimeError("Cannot run operation as ssh key is missing")
    return ssh_key_path.read_text("utf-8")


def dep_database_dir(settings: Annotated[Settings, Depends(dep_settings)]) -> Path:
    return settings.SERVER_START_PATH.expanduser().resolve()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


async def verify_token(
    token: Annotated[str, Depends(oauth2_scheme)], settings: Annotated[Settings, Depends(dep_settings)]
) -> None:
    if settings.JWT_SECRET is None:
        return

    def _http_exception(detail: str) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"}, detail=detail
        )

    try:
        decoded = decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        if decoded["deployment_name"] != settings.DEPLOYMENT_NAME:
            raise _http_exception(detail="Cannot access deployment")
    except BaseException:
        raise _http_exception(detail="Cannot access deployment")


async def verify_premium(settings: Annotated[Settings, Depends(dep_settings)]):
    if not settings.ENV != Env.PACKAGED:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Search is not enabled")
