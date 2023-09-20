import math
import uuid
from email.utils import parsedate
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Set, Tuple, Union

import encord
from cachetools import TTLCache, cached
from encord.http.constants import RequestsSettings
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlmodel import Session, select
from starlette.responses import FileResponse, Response
from starlette.staticfiles import NotModifiedResponse

from encord_active.db.enums import (
    AnnotationEnums,
    DataEnums,
    DataType,
    EnumDefinition,
    EnumType,
)
from encord_active.db.local_data import db_uri_to_local_file_path
from encord_active.db.metrics import (
    AnnotationMetrics,
    DataMetrics,
    MetricDefinition,
    MetricType,
)
from encord_active.db.models import (
    AnnotationType,
    EmbeddingReductionType,
    Project,
    ProjectAnnotationAnalytics,
    ProjectCollaborator,
    ProjectDataAnalytics,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectEmbeddingReduction,
    ProjectPrediction,
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
)
from encord_active.db.queries.tags import (
    select_frame_data_tags,
    select_frame_label_tags,
)
from encord_active.imports.sandbox.sandbox_projects import available_sandbox_projects
from encord_active.server.dependencies import (
    DataItem,
    dep_database_dir,
    dep_engine,
    dep_engine_readonly,
    dep_settings,
    dep_ssh_key,
    parse_data_item,
)
from encord_active.server.routers import (
    project2_actions,
    project2_analysis,
    project2_prediction,
    project2_tags,
)
from encord_active.server.routers.project2_prediction import AnnotationEnumItem
from encord_active.server.routers.queries.metric_query import sql_count
from encord_active.server.settings import Settings

router = APIRouter(
    prefix="/projects_v2",
    tags=["projects_v2"],
)
router.include_router(project2_analysis.router)
router.include_router(project2_prediction.router)
router.include_router(project2_actions.router)
router.include_router(project2_tags.router)


class ProjectSearchEntry(BaseModel):
    project_hash: uuid.UUID
    title: str
    description: str
    sandbox: bool


class ProjectSandboxEntry(BaseModel):
    project_hash: uuid.UUID
    title: str
    description: str

    sandbox_url: str
    data_count: int
    annotation_count: int
    class_count: int


class ProjectSearchResult(BaseModel):
    projects: List[ProjectSearchEntry]
    sandbox_projects: List[ProjectSandboxEntry]


@router.get("")
def route_list_projects(
    engine: Engine = Depends(dep_engine_readonly), settings: Settings = Depends(dep_settings)
) -> ProjectSearchResult:
    # Load existing projects
    projects: List[ProjectSearchEntry] = []
    project_dict: Dict[uuid.UUID, ProjectSearchEntry] = {}
    with Session(engine) as sess:
        project_res = sess.exec(
            select(Project.project_hash, Project.name, Project.description).order_by(
                Project.name,
            )
        ).fetchall()
        for project_hash, title, description in project_res:
            entry = ProjectSearchEntry(project_hash=project_hash, title=title, description=description, sandbox=False)
            projects.append(entry)
            project_dict[project_hash] = entry

    # Load sandbox projects
    sandbox_projects: List[ProjectSandboxEntry] = []
    for name, data in available_sandbox_projects(settings.AVAILABLE_SANDBOX_PROJECTS).items():
        project_hash = data.hash
        if project_hash in project_dict:
            project_dict[project_hash].sandbox = True
            continue
        sandbox_projects.append(
            ProjectSandboxEntry(
                project_hash=project_hash,
                title=data.name,
                description="",
                sandbox_url=f"/ea-sandbox-static/{data.image_filename}",
                data_count=data.stats.data_hash_count,
                annotation_count=data.stats.annotation_count,
                class_count=data.stats.class_count,
            )
        )
    sandbox_projects.sort(key=lambda s: s.title)

    return ProjectSearchResult(projects=projects, sandbox_projects=sandbox_projects)


class ProjectMetadata(BaseModel):
    data_count: int
    annotation_count: int
    class_count: int


@router.get("/{project_hash}/metadata")
@cached(cache=TTLCache(maxsize=1, ttl=60 * 5))  # 5 minute TTL Cache
def route_project_metadata(project_hash: uuid.UUID, engine: Engine = Depends(dep_engine_readonly)) -> ProjectMetadata:
    with Session(engine) as sess:
        ontology = sess.exec(select(Project.ontology).where(Project.project_hash == project_hash)).first()
        if ontology is None:
            raise HTTPException(status_code=404, detail="Project does not exist")
        data_count = sess.exec(select(sql_count()).where(ProjectDataMetadata.project_hash == project_hash)).first()
        annotation_count = sess.exec(
            select(sql_count()).where(ProjectAnnotationAnalytics.project_hash == project_hash)
        ).first()
        preview_by_annotate = sess.exec(
            select(ProjectAnnotationAnalytics.du_hash, ProjectAnnotationAnalytics.frame)
            .where(ProjectAnnotationAnalytics.project_hash == project_hash)
            .order_by(ProjectAnnotationAnalytics.du_hash, ProjectAnnotationAnalytics.frame)
            .limit(1)
        ).first()
        if preview_by_annotate is None:
            preview_by_annotate = sess.exec(
                select(ProjectDataAnalytics.du_hash, ProjectDataAnalytics.frame)
                .where(ProjectDataAnalytics.project_hash == project_hash)
                .order_by(ProjectDataAnalytics.du_hash, ProjectDataAnalytics.frame)
                .limit(1)
            ).first()

        if preview_by_annotate is None:
            raise HTTPException(status_code=404, detail="Project contains no data")
        preview_du_hash, preview_frame = preview_by_annotate

    return ProjectMetadata(
        data_count=data_count,
        annotation_count=annotation_count,
        class_count=len(ontology["objects"]) + len(ontology["classifications"]),
        # preview_du_hash=preview_du_hash,
        # preview_frame=preview_frame,
    )


class MetricSummary(BaseModel):
    title: str
    short_desc: str
    long_desc: str
    type: MetricType


def _metric_summary(
    metrics: Dict[str, MetricDefinition], fixme_exclude: Optional[Set[str]] = None
) -> Dict[str, MetricSummary]:
    return {
        metric_name: MetricSummary(
            title=metric.title,
            short_desc=metric.short_desc,
            long_desc=metric.long_desc,
            type=metric.type,
        )
        for metric_name, metric in metrics.items()
        if fixme_exclude is None or metric_name not in fixme_exclude
    }


class EnumSummary(BaseModel):
    title: str
    values: Optional[dict[str, str]]
    type: EnumType


def _enum_summary(enums: Dict[str, EnumDefinition]) -> Dict[str, EnumSummary]:
    return {
        enum_name: EnumSummary(
            title=enum.title,
            values=enum.values,
            type=enum.enum_type,
        )
        for enum_name, enum in enums.items()
    }


class ProjectDomainSummary(BaseModel):
    metrics: Dict[str, MetricSummary]
    enums: Dict[str, EnumSummary]


class ProjectSummary(BaseModel):
    name: str
    description: str
    ontology: dict
    local_project: bool
    data: ProjectDomainSummary
    annotation: ProjectDomainSummary
    du_count: int
    frame_count: int
    annotation_count: int
    classification_count: int
    data_annotation: ProjectDomainSummary
    tags: Dict[uuid.UUID, str]
    preview: Optional[str]


@router.get("/{project_hash}/summary")
def route_project_summary(project_hash: uuid.UUID, engine: Engine = Depends(dep_engine_readonly)) -> ProjectSummary:
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError()

        tags = sess.exec(select(ProjectTag).where(ProjectTag.project_hash == project_hash)).fetchall()

        preview = sess.exec(
            select(ProjectDataAnalytics.du_hash, ProjectDataAnalytics.frame)
            .where(ProjectDataAnalytics.project_hash == project_hash)
            .order_by(
                ProjectDataAnalytics.metric_object_count.desc(),  # type: ignore
                ProjectDataAnalytics.du_hash,
                ProjectDataAnalytics.frame,
            )
            .limit(1)
        ).first()
        if preview is None:
            preview = sess.exec(
                select(ProjectDataUnitMetadata.du_hash, ProjectDataUnitMetadata.frame)
                .where(ProjectDataUnitMetadata.project_hash == project_hash)
                .order_by(ProjectDataUnitMetadata.du_hash, ProjectDataUnitMetadata.frame)
                .limit(1)
            ).first()

        du_count = sess.exec(
            select(sql_count())
            .where(ProjectDataAnalytics.project_hash == project_hash)
            .group_by(ProjectDataAnalytics.du_hash)
        ).first()
        frame_count = sess.exec(
            select(sql_count())
            .where(ProjectDataAnalytics.project_hash == project_hash)
            .group_by(ProjectDataAnalytics.du_hash)
        ).first()
        annotation_count = sess.exec(
            select(sql_count()).where(ProjectAnnotationAnalytics.project_hash == project_hash)
        ).first()
        classification_count = sess.exec(
            select(sql_count()).where(
                ProjectAnnotationAnalytics.project_hash == project_hash,
                ProjectAnnotationAnalytics.annotation_type == AnnotationType.CLASSIFICATION,
            )
        ).first()

    return ProjectSummary(
        name=project.name,
        description=project.description,
        ontology=project.ontology,
        local_project=not project.remote,
        data=ProjectDomainSummary(
            metrics=_metric_summary(DataMetrics),
            enums=_enum_summary(DataEnums),
        ),
        annotation=ProjectDomainSummary(
            metrics=_metric_summary(
                AnnotationMetrics,
            ),
            enums=_enum_summary(AnnotationEnums),
        ),
        du_count=du_count,
        frame_count=frame_count,
        annotation_count=annotation_count,
        classification_count=classification_count,
        data_annotation=ProjectDomainSummary(
            metrics=_metric_summary(AnnotationMetrics | DataMetrics),
            enums=_enum_summary(AnnotationEnums | DataEnums),
        ),
        tags={tag.tag_hash: tag.name for tag in tags},
        preview=f"{preview[0]}_{preview[1]}" if preview is not None else None,
    )


class ProjectList2DEmbeddingReductionResultEntry(BaseModel):
    name: str
    description: str
    hash: uuid.UUID
    type: EmbeddingReductionType


class ProjectList2DEmbeddingReductionResult(BaseModel):
    total: int
    results: List[ProjectList2DEmbeddingReductionResultEntry]


@router.get("/{project_hash}/reductions")
def route_project_list_reductions(
    project_hash: uuid.UUID, engine: Engine = Depends(dep_engine_readonly)
) -> ProjectList2DEmbeddingReductionResult:
    with Session(engine) as sess:
        r = sess.exec(
            select(ProjectEmbeddingReduction).where(ProjectEmbeddingReduction.project_hash == project_hash)
        ).fetchall()
    return ProjectList2DEmbeddingReductionResult(
        total=len(r),
        results=[
            ProjectList2DEmbeddingReductionResultEntry(
                name=e.reduction_name, description=e.reduction_description, hash=e.reduction_hash, type=e.reduction_type
            )
            for e in r
        ],
    )


@router.get("/{project_hash}/files/{du_hash}/{frame}")
def route_project_raw_file(
    project_hash: uuid.UUID,
    du_hash: uuid.UUID,
    frame: int,
    if_none_match: Annotated[Optional[str], Header()] = None,
    if_modified_since: Annotated[Optional[str], Header()] = None,
    engine: Engine = Depends(dep_engine_readonly),
    database_dir: Path = Depends(dep_database_dir),
) -> Response:
    with Session(engine) as sess:
        query = select(ProjectDataUnitMetadata.data_uri,).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
            ProjectDataUnitMetadata.du_hash == du_hash,
            ProjectDataUnitMetadata.frame == frame,
        )
        data_uri = sess.exec(query).first()
        if data_uri is not None:
            url_path = db_uri_to_local_file_path(data_uri, database_dir)
            if url_path is not None:
                file_response = FileResponse(url_path, method="GET")
                etag = file_response.headers.get("etag", None)
                if if_none_match is not None and etag is not None and if_none_match == etag:
                    return NotModifiedResponse(file_response.headers)
                if if_modified_since is not None:
                    last_modified = file_response.headers.get("last-modified", None)
                    if last_modified is not None:
                        if_modified_since_date = parsedate(if_modified_since)
                        last_modified_date = parsedate(last_modified)
                        if (
                            if_modified_since_date is not None
                            and last_modified_date is not None
                            and if_modified_since_date >= last_modified_date
                        ):
                            return NotModifiedResponse(file_response.headers)
                return file_response
        raise HTTPException(status_code=404, detail="Project local file not found")


@cached(cache=TTLCache(maxsize=1, ttl=60 * 5))  # 60 * 5 (s) = 5 min
def _cache_encord_user_client(ssh_key: str) -> encord.EncordUserClient:
    return encord.EncordUserClient.create_with_ssh_private_key(
        ssh_key,
        requests_settings=RequestsSettings(max_retries=5),
    )


@cached(cache=TTLCache(maxsize=50, ttl=60 * 5))  # 60 * 5 (s) = 5 min
def _cache_encord_project(ssh_key: str, project_hash: uuid.UUID) -> encord.Project:
    return _cache_encord_user_client(ssh_key).get_project(str(project_hash))


@cached(cache=TTLCache(maxsize=50_000, ttl=60 * 30))  # 60 * 30 (s) = 30 min
def _cache_encord_img_lookup(
    ssh_key: str,
    project_hash: uuid.UUID,
    data_hash: uuid.UUID,
) -> Dict[uuid.UUID, str]:
    encord_project = _cache_encord_project(ssh_key, project_hash)
    video, images = encord_project.get_data(str(data_hash), get_signed_url=True)
    encord_url_dict = {}
    for image in images or []:
        encord_url_dict[uuid.UUID(image["data_hash"])] = str(image["file_link"])
    if video is not None:
        encord_url_dict[data_hash] = str(video["file_link"])
    return encord_url_dict


class ProjectItemTags(BaseModel):
    data: List[ProjectTag]
    label: Dict[str, List[ProjectTag]]


class DataEnumItem(BaseModel):
    data_type: DataType


class ProjectItem(BaseModel):
    data_metrics: Dict[str, Optional[float]]
    annotation_metrics: Dict[str, Dict[str, Optional[float]]]
    data_enums: DataEnumItem
    annotation_enums: Dict[str, AnnotationEnumItem]
    objects: list[dict]
    classifications: list[dict]
    object_answers: Dict[str, dict]
    classification_answers: Dict[str, dict]
    dataset_title: str
    dataset_hash: uuid.UUID
    data_title: str
    data_hash: uuid.UUID
    label_hash: uuid.UUID
    num_frames: int
    frames_per_second: Optional[float]
    data_type: str
    url: str
    timestamp: Optional[float]
    tags: ProjectItemTags


def _sanitise_nan(value: Optional[float]) -> Optional[float]:
    if value is None or math.isnan(value):
        return None
    return value


@router.get("/{project_hash}/item/{data_item}/")
def route_project_data_item(
    project_hash: uuid.UUID,
    item_details: DataItem = Depends(parse_data_item),
    engine: Engine = Depends(dep_engine_readonly),
    ssh_key: str = Depends(dep_ssh_key),
    database_dir: Path = Depends(dep_database_dir),
) -> ProjectItem:
    du_hash = item_details.du_hash
    frame = item_details.frame
    with Session(engine) as sess:
        project_remote = sess.exec(select(Project.remote).where(Project.project_hash == project_hash)).first()
        du_meta = sess.exec(
            select(ProjectDataUnitMetadata).where(
                ProjectDataUnitMetadata.project_hash == project_hash,
                ProjectDataUnitMetadata.du_hash == du_hash,
                ProjectDataUnitMetadata.frame == frame,
            )
        ).first()
        if du_meta is None:
            raise ValueError(f"Invalid request: du_hash={du_hash}, frame={frame} is missing DataUnitMeta")
        data_meta = sess.exec(
            select(ProjectDataMetadata).where(
                ProjectDataMetadata.project_hash == project_hash, ProjectDataMetadata.data_hash == du_meta.data_hash
            )
        ).first()
        if data_meta is None:
            raise ValueError(f"Invalid request: du_hash={du_hash}, frame={frame} is missing DataMeta")
        du_analytics = sess.exec(
            select(ProjectDataAnalytics).where(
                ProjectDataAnalytics.project_hash == project_hash,
                ProjectDataAnalytics.du_hash == du_hash,
                ProjectDataAnalytics.frame == frame,
            )
        ).first()
        annotation_analytics_list = sess.exec(
            select(ProjectAnnotationAnalytics).where(
                ProjectAnnotationAnalytics.project_hash == project_hash,
                ProjectAnnotationAnalytics.du_hash == du_hash,
                ProjectAnnotationAnalytics.frame == frame,
            )
        ).fetchall()

        data_tags = sess.exec(select_frame_data_tags(project_hash, du_hash, frame)).fetchall()
        label_tags: Dict[str, List[ProjectTag]] = {}
        for tag, annotation_hash in sess.exec(select_frame_label_tags(project_hash, du_hash, frame)).fetchall():
            label_tags.setdefault(annotation_hash, []).append(tag)

    if du_meta is None or du_analytics is None:
        raise ValueError

    uri = du_meta.data_uri
    uri_is_video = du_meta.data_uri_is_video
    frames_per_second = data_meta.frames_per_second
    if uri is not None:
        url_path = db_uri_to_local_file_path(uri, database_dir)
        if url_path is not None:
            uri = f"/api/projects_v2/{project_hash}/files/{du_hash}/{frame}"
    elif project_remote:
        uri = _cache_encord_img_lookup(
            ssh_key=ssh_key,
            project_hash=project_hash,
            data_hash=du_meta.data_hash,
        )[du_hash]
    else:
        raise ValueError(f"Cannot resolve project url: {project_hash} / {du_hash}")

    timestamp = None
    if uri_is_video:
        if frames_per_second is None:
            raise ValueError("Video defined but missing valid frames_per_second")
        timestamp = (float(int(frame)) + 0.5) / float(frames_per_second)

    all_objects = {obj["objectHash"] for obj in du_meta.objects}
    all_classifications = {cls["classificationHash"] for cls in du_meta.classifications}

    return ProjectItem(
        data_metrics={k: _sanitise_nan(getattr(du_analytics, k)) for k in DataMetrics},
        annotation_metrics={
            annotation_analytics.annotation_hash: {
                k: _sanitise_nan(getattr(annotation_analytics, k)) for k in AnnotationMetrics
            }
            for annotation_analytics in annotation_analytics_list
        },
        data_enums={k: getattr(du_analytics, k) for k in DataEnums if k != "tags"},
        annotation_enums={
            annotation_analytics.annotation_hash: {
                k: getattr(annotation_analytics, k) for k in AnnotationEnums if k != "tags"
            }
            for annotation_analytics in annotation_analytics_list
        },
        objects=du_meta.objects,
        classifications=du_meta.classifications,
        object_answers={k: v for k, v in data_meta.object_answers.items() if k in all_objects},
        classification_answers={k: v for k, v in data_meta.classification_answers.items() if k in all_classifications},
        dataset_title=data_meta.dataset_title,
        dataset_hash=data_meta.dataset_hash,
        data_title=data_meta.data_title,
        data_hash=data_meta.data_hash,
        label_hash=data_meta.label_hash,
        num_frames=data_meta.num_frames,
        frames_per_second=data_meta.frames_per_second,
        data_type=data_meta.data_type,
        url=uri,
        timestamp=timestamp,
        tags=ProjectItemTags(data=data_tags, label=label_tags),
    )


class ListProjectPredictionResultEntry(BaseModel):
    name: str
    prediction_hash: uuid.UUID


class ListProjectPredictionResult(BaseModel):
    total: int
    results: List[ListProjectPredictionResultEntry]


@router.get("/{project_hash}/predictions")
def route_project_list_predictions(
    project_hash: uuid.UUID,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    order_by: Optional[Literal[""]] = None,
    engine: Engine = Depends(dep_engine_readonly),
) -> ListProjectPredictionResult:
    with Session(engine) as sess:
        predictions = sess.exec(
            select(ProjectPrediction).where(ProjectPrediction.project_hash == project_hash).offset(offset).limit(limit)
        ).fetchall()
    return ListProjectPredictionResult(
        total=len(predictions),
        results=[
            ListProjectPredictionResultEntry(name=prediction.name, prediction_hash=prediction.prediction_hash)
            for prediction in predictions
        ],
    )


class ProjectCollaboratorEntry(BaseModel):
    email: str
    id: int


@router.get("/{project_hash}/collaborators")
def route_project_list_collaborators(
    project_hash: uuid.UUID,
    engine: Engine = Depends(dep_engine_readonly),
) -> List[ProjectCollaboratorEntry]:
    with Session(engine) as sess:
        collaborators = sess.exec(
            select(ProjectCollaborator).where(ProjectCollaborator.project_hash == project_hash)
        ).fetchall()
    return [ProjectCollaboratorEntry(email=c.user_email, id=c.user_id) for c in collaborators]


@router.post("/{project_hash}/create/tag")
def route_project_action_tag_items(
    project_hash: uuid.UUID, tagged_items: List[str], tag_name: str, engine: Engine = Depends(dep_engine)
) -> None:
    tag_hash = uuid.uuid4()
    data: List[Tuple[uuid.UUID, int]] = []
    annotations: List[Tuple[uuid.UUID, int, str]] = []
    with Session(engine) as sess:
        sess.add(ProjectTag(tag_hash=tag_hash, project_hash=project_hash, name=tag_name))
        tags = [
            ProjectTaggedDataUnit(project_hash=project_hash, du_hash=du_hash, frame=frame, tag_hash=tag_hash)
            for du_hash, frame in data
        ]
        sess.add_all(tags)
        sess.commit()

    with Session(engine) as sess:
        sess.add(ProjectTag(tag_hash=tag_hash, project_hash=project_hash, name=tag_name))
        tags2 = [
            ProjectTaggedAnnotation(
                project_hash=project_hash,
                du_hash=du_hash,
                frame=frame,
                annotation_hash=annotation_hash,
                tag_hash=tag_hash,
            )
            for du_hash, frame, annotation_hash in annotations
        ]
        sess.add_all(tags2)
        sess.commit()
