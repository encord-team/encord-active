from os import environ
from typing import Optional, Dict, Tuple, Union, List
from urllib.parse import quote

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from encord_active.db.models import get_engine, Project, ProjectDataUnitMetadata, ProjectTaggedDataUnit, \
    ProjectTaggedObject, ProjectTag, ProjectDataAnalytics, ProjectObjectAnalytics
from encord_active.lib.common.data_utils import url_to_file_path
from encord_active.server.dependencies import verify_token
from encord_active.server.settings import get_settings

router = APIRouter(
    prefix="/projects_v2",
    tags=["projects_v2"],
    dependencies=[Depends(verify_token)],
)

engine_path = get_settings().SERVER_START_PATH / "encord-active.sqlite"
engine = get_engine(engine_path)


@router.get("/list")
def get_all_projects():
    with Session(engine) as sess:
        projects = sess.exec(
            select(Project.project_hash, Project.project_name, Project.project_description)
        ).fetchall()
    return {
        project_hash: {
            "title": title,
            "description": description,
        }
        for project_hash, title, description in projects
    }


@router.get("/get/{project_hash}/summary")
def get_project_summary():
    with Session(engine) as sess:
        pass

    return {
        "title": "",
        "name": "",
        "ontology": "",
        "data_metrics": [],
        "label_metrics": [],
        "tags": [],
    }


@router.get("/get/{project_hash}/preview/{du_hash}/{frame}/{object_hash?}")
def display_preview(project_hash: str, du_hash: str, frame: int, object_hash: Optional[str] = None):
    with Session(engine) as sess:
        query = select(
            ProjectDataUnitMetadata.data_uri,
            ProjectDataUnitMetadata.data_uri_is_video,
            ProjectDataUnitMetadata.objects
        ).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
            ProjectDataUnitMetadata.du_hash == du_hash,
            ProjectDataUnitMetadata.frame == frame
        )
        result = sess.exec(query).fetchone()
        if object_hash is None:
            query_tags = select(
                ProjectTag,
            ).where(
                ProjectTaggedDataUnit.project_hash == project_hash,
                ProjectTaggedDataUnit.du_hash == du_hash,
                ProjectTaggedDataUnit.frame == frame,
            ).join(ProjectTaggedDataUnit, ProjectTaggedDataUnit.tag_hash == ProjectTag.tag_hash)
        else:
            query_tags = select(
                ProjectTag,
            ).where(
                ProjectTaggedObject.project_hash == project_hash,
                ProjectTaggedObject.du_hash == du_hash,
                ProjectTaggedObject.frame == frame,
                ProjectTaggedObject.object_hash == object_hash,
            ).join(ProjectTaggedObject, ProjectTaggedObject.tag_hash == ProjectTag.tag_hash)
        result_tags = sess.exec(query_tags).fetchall()
    if result is None:
        return None
    uri, uri_is_video, objects = result
    if object_hash is None:
        objects = [
            obj
            for obj in objects
            if obj["objectHash"] == object_hash
        ]
        if len(objects) == 0:
            return None
    if uri is not None:
        settings = get_settings()
        relative_path = settings.SERVER_START_PATH
        # FIXME:
        raise ValueError("Not yet implemented")
        # url_path = url_to_file_path(uri, project_dir) # FIXME: store project_dir in database.
        # if url_path is not None:
        #    uri = f"{settings.API_URL}/ea-static/{quote(relative_path.as_posix())}"
    else:
        # Also: NOT YET IMPLEMENTED!!!
        pass
    return {
        "url": uri,
        "video_timestamp": "",
        "objects": objects,
        "tags": [
            {
                "name": tag.name,
                "tag_hash": tag.tag_hash
            }
            for tag in result_tags
        ]
    }


@router.get("/get/{project_hash}/data_search")
def search_data(
        project_hash: str,
        metric_filters: Dict[str, Tuple[Union[int, float], Union[int, float]]],
        enum_filters: Dict[str, List[str]],
        order_by: Optional[str] = None,
        desc: bool = False
):
    # FIXME: Need hardcoded definition of all metrics for the 2 queries
    query_filters = []

    with Session(engine) as sess:
        search_query = select(
            ProjectDataAnalytics.du_hash,
            ProjectDataAnalytics.frame,
        ).where(
            ProjectDataAnalytics.project_hash == project_hash,
            *query_filters
        ).limit(
            # 10_000 max results in a search query (+1 to detect truncation).
            limit=10_001
        )
        search_results = sess.exec(search_query).fetchall()
    truncated = len(search_results) == 10_001

    return {
        "truncated": truncated,
        "results": [
            {"du_hash": du_hash, "frame": frame}
            for du_hash, frame in search_results
        ]
    }


@router.get("/get/{project_hash}/label_search")
def search_labels(project_hash: str, filters: Dict[str, Tuple[int, int]]) -> None:
    # FIXME: search (new tables)
    return None


@router.get("/get/{project_hash}/data_scatter2d")
def scatter_2d_data_metric(project_hash: str, x_metric: str, y_metric: str):
    with Session(engine) as sess:
        scatter_query = select(
            ProjectDataAnalytics.du_hash,
            ProjectDataAnalytics.frame,
            x_metric,
            y_metric  # FIXME: resolve to actual keys
        ).where(ProjectDataAnalytics.project_hash == project_hash)
        scatter_results = sess.exec(scatter_query).fetchall()
    return [
        {"x": x, "y": y, "du_hash": du_hash, "frame": frame}
        for du_hash, frame, x, y in scatter_results
    ]


@router.get("/get/{project_hash}/label_scatter2d")
def scatter_2d_label_metric(project_hash: str, x_metric: str, y_metric: str):
    with Session(engine) as sess:
        scatter_query = select(
            ProjectObjectAnalytics.du_hash,
            ProjectObjectAnalytics.frame,
            x_metric,
            y_metric  # FIXME: resolve to actual keys
        ).where(ProjectObjectAnalytics.project_hash == project_hash)
        scatter_results = sess.exec(scatter_query).fetchall()
    return [
        {"x": x, "y": y, "du_hash": du_hash, "frame": frame}
        for du_hash, frame, x, y in scatter_results
    ]


@router.get("/get/{project_hash}")
@router.get("/get/{project_hash}/metric_outliers")
def get_outliers(project_hash, metric_name: str) -> None:
    # FIXME: list all outliers (for outlier summary)
    return None
