from typing import Optional, Dict

from fastapi import APIRouter, Depends

from encord_active.server.dependencies import verify_token

router = APIRouter(
    prefix="/projects2",
    tags=["projects2"],
    dependencies=[Depends(verify_token)],
)


@router.get("/{project_hash}/preview/{data_hash}/{frame}/{object_hash?}")
def display_preview(project_hash: str, data_hash: str, frame: int, object_hash: Optional[str] = None) -> None:
    # FIXME: (resolve url + labels):
    return None


@router.get("/{project_hash}/detail/{data_hash}/{frame}/{object_hash?}")
def display_detailed(project_hash: str, data_hash: str, frame: int, object_hash: Optional[str] = None) -> None:
    # FIXME: (resolve url + labels + extra metadata: metrics + embeddings).
    return None


def search(project_hash: str, filters: Dict[str, str]) -> None:
    # FIXME: search (new tables)
    return None


def scatter_2d_metric(project_hash: str, data_hash: str) -> None:
    # FIXME: internal comparison (used for img width x img height)
    # any metric (including image dimensions)
    return None

@router.get("/{project_hash/{metric_name}")
def get_outliers(project_hash, metric_name: str) -> None:
    # FIXME: list all outliers (for outlier summary)
    return None


@router.get("/{project_hash}/summary")
def project_summary(project_hash: str) -> None:
    # FIXME: all project metadata for the initial summary
    pass

