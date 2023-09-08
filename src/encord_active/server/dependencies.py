import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import decode
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from .settings import Env, get_settings


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


def parse_data_or_annotate_item(item: str) -> DataOrAnnotateItem:
    segments = item.split("_")
    if len(segments) > 3 or len(segments) < 2:
        raise ValueError(f"Item expects 2 segments: {item}")
    du_hash, frame = segments[:2]
    annotation_hash = None if len(segments) == 2 else segments[2]
    if annotation_hash is not None and len(annotation_hash) != 8:
        raise ValueError(f"Item annotation_hash should be exactly 8 characters: {annotation_hash}")
    return DataOrAnnotateItem(du_hash=uuid.UUID(du_hash), frame=int(frame), annotation_hash=annotation_hash)


def dep_engine() -> Engine:
    raise RuntimeError("Missing Engine")


def dep_oauth2_scheme() -> OAuth2PasswordBearer:
    raise RuntimeError("Missing OAuth2PasswordBearer")


def dep_ssh_key() -> str:
    raise RuntimeError("Missing ssh_key")


def dep_database_dir() -> Path:
    settings = get_settings()
    return settings.SERVER_START_PATH.expanduser().resolve()


async def verify_token(token: Annotated[str, Depends(dep_oauth2_scheme)]) -> None:
    settings = get_settings()
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


async def verify_premium():
    if not get_settings().ENV != Env.PACKAGED:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Search is not enabled")
