import uuid
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import decode
from pydantic import BaseModel
from sqlmodel import Session, select

from encord_active.db.models import Project, get_engine
from encord_active.lib.project.project_file_structure import ProjectFileStructure

from .settings import Env, get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

engine_path = get_settings().SERVER_START_PATH / "encord-active.sqlite"
engine = get_engine(engine_path, concurrent=True)


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
        raise ValueError(f"Item annotation_hash should be exactly 8 characters")
    return DataOrAnnotateItem(du_hash=uuid.UUID(du_hash), frame=int(frame), annotation_hash=annotation_hash)


@lru_cache
def _try_find_project(path: Path, name: str, hash: str):
    direct_match = path / name.lower().replace(" ", "-")
    if direct_match.is_dir():
        return ProjectFileStructure(direct_match)

    for pfs in [ProjectFileStructure(path) for path in path.glob("*") if path.is_dir()]:
        if pfs.project_meta.exists() and pfs.load_project_meta()["project_hash"] == hash:
            return pfs

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found on disk.")


async def get_project_file_structure(project: str) -> ProjectFileStructure:
    if (get_settings().SERVER_START_PATH / project).exists():
        return ProjectFileStructure(get_settings().SERVER_START_PATH / project)

    with Session(engine) as sess:
        db_project = sess.exec(select(Project).where(Project.project_hash == uuid.UUID(project))).first()
        if not db_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Project: {project} wasn't found in the DB"
            )

        return _try_find_project(
            get_settings().SERVER_START_PATH, db_project.project_name, str(db_project.project_hash)
        )


ProjectFileStructureDep = Annotated[ProjectFileStructure, Depends(get_project_file_structure)]


async def verify_token(token: Annotated[str, Depends(oauth2_scheme)]) -> None:
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
    except:
        raise _http_exception(detail="Cannot access deployment")


async def verify_token_with_project_hash(token: Annotated[str, Depends(oauth2_scheme)], project_hash: str) -> None:
    # FIXME: tokens should give information about which project_hashes are allowed.
    return await verify_token(token)


async def verify_premium():
    if not get_settings().ENV != Env.PACKAGED:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Search is not enabled")
