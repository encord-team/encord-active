from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import decode

from encord_active.lib.project.project_file_structure import ProjectFileStructure

from .settings import Env, get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


async def get_project_file_structure(project: str) -> ProjectFileStructure:
    return ProjectFileStructure(get_settings().SERVER_START_PATH / project)


ProjectFileStructureDep = Annotated[ProjectFileStructure, Depends(get_project_file_structure)]


async def verify_token(token: Annotated[str, Depends(oauth2_scheme)]):
    settings = get_settings()
    if settings.JWT_SECRET is None:
        return

    try:
        decoded = decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        if decoded["deployment_name"] != settings.DEPLOYMENT_NAME:
            raise ValueError()
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_premium():
    if not get_settings().ENV != Env.LOCAL:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Search is not enabled")
