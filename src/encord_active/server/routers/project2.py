import uuid
from typing import Dict, List, Literal, Optional, Set

from cachetools import TTLCache, cached
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlmodel import Session, select
from starlette.responses import FileResponse

from encord_active.db.enums import AnnotationEnums, DataEnums, EnumDefinition, EnumType
from encord_active.db.metrics import AnnotationMetrics, DataMetrics, MetricDefinition, MetricType
from encord_active.db.models import (
    AnnotationType,
    EmbeddingReductionType,
    Project,
    ProjectAnnotationAnalytics,
    ProjectDataAnalytics,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectEmbeddingReduction,
    ProjectPrediction,
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
)
from encord_active.lib.common.data_utils import url_to_file_path
from encord_active.lib.encord.utils import get_encord_project
from encord_active.lib.project.sandbox_projects.sandbox_projects import (
    available_prebuilt_projects,
)
from encord_active.server.dependencies import (
    DataOrAnnotateItem,
    parse_data_or_annotate_item,
    DataItem,
    parse_data_item,
    dep_engine,
)
from encord_active.server.routers import (
    project2_actions,
    project2_analysis,
    project2_prediction,
)
from encord_active.server.routers.queries.metric_query import sql_count
from encord_active.server.settings import get_settings

router = APIRouter(
    prefix="/projects_v2",
    tags=["projects_v2"],
)
router.include_router(project2_analysis.router)
router.include_router(project2_prediction.router)
router.include_router(project2_actions.router)


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
def get_all_projects(engine: Engine = Depends(dep_engine)) -> ProjectSearchResult:
    # Load existing projects
    projects: List[ProjectSearchEntry] = []
    project_dict: Dict[uuid.UUID, ProjectSearchEntry] = {}
    with Session(engine) as sess:
        project_res = sess.exec(
            select(Project.project_hash, Project.project_name, Project.project_description).order_by(
                Project.project_name,
            )
        ).fetchall()
        for project_hash, title, description in project_res:
            entry = ProjectSearchEntry(project_hash=project_hash, title=title, description=description, sandbox=False)
            projects.append(entry)
            project_dict[project_hash] = entry

    # Load sandbox projects
    sandbox_projects: List[ProjectSandboxEntry] = []
    for name, data in available_prebuilt_projects(get_settings().AVAILABLE_SANDBOX_PROJECTS).items():
        project_hash = uuid.UUID(data["hash"])
        if project_hash in project_dict:
            project_dict[project_hash].sandbox = True
            continue
        sandbox_projects.append(
            ProjectSandboxEntry(
                project_hash=project_hash,
                title=name,
                description="",
                sandbox_url=f"/ea-sandbox-static/{data['image_filename']}",
                data_count=data["stats"]["dataUnits"],
                annotation_count=data["stats"]["labels"],
                class_count=data["stats"]["classes"],
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
def get_project_metadata(project_hash: uuid.UUID, engine: Engine = Depends(dep_engine)) -> ProjectMetadata:
    with Session(engine) as sess:
        ontology = sess.exec(select(Project.project_ontology).where(Project.project_hash == project_hash)).first()
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
        enum_name: {
            "title": enum.title,
            "values": enum.values,
            "type": enum.enum_type,
        }
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
def get_project_summary(project_hash: uuid.UUID, engine: Engine = Depends(dep_engine)) -> ProjectSummary:
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError()

        tags = sess.exec(select(ProjectTag).where(ProjectTag.project_hash == project_hash)).fetchall()

        preview = sess.exec(
            select(ProjectAnnotationAnalytics.du_hash, ProjectAnnotationAnalytics.frame)
            .where(ProjectAnnotationAnalytics.project_hash == project_hash)
            .limit(1)
        ).first()
        if preview is None:
            preview = sess.exec(
                select(ProjectDataUnitMetadata.du_hash, ProjectDataUnitMetadata.frame)
                .where(ProjectDataUnitMetadata.project_hash == project_hash)
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
        name=project.project_name,
        description=project.project_description,
        ontology=project.project_ontology,
        local_project=project.project_remote_ssh_key_path is None,
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
def list_supported_2d_embedding_reductions(
    project_hash: uuid.UUID, engine: Engine = Depends(dep_engine)
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
def display_raw_file(
    project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int, engine: Engine = Depends(dep_engine)
) -> FileResponse:
    with Session(engine) as sess:
        query = select(ProjectDataUnitMetadata.data_uri,).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
            ProjectDataUnitMetadata.du_hash == du_hash,
            ProjectDataUnitMetadata.frame == frame,
        )
        data_uri = sess.exec(query).first()
        if data_uri is not None:
            settings = get_settings()
            root_path = settings.SERVER_START_PATH.expanduser().resolve()
            url_path = url_to_file_path(data_uri, root_path)
            if url_path is not None:
                # FIXME: add not-modified case response (see StaticFiles)
                return FileResponse(url_path)
        raise HTTPException(status_code=404, detail="Project local file not found")


@cached(cache=TTLCache(maxsize=10_000, ttl=60 * 30))  # 60* 30 (s) = 30 mins
def _cache_encord_img_lookup(
    ssh_key_path: str,
    project_hash: uuid.UUID,
    data_hash: uuid.UUID,
) -> Dict[uuid.UUID, str]:
    encord_project = get_encord_project(ssh_key_path=ssh_key_path, project_hash=str(project_hash))
    video, images = encord_project.get_data(str(data_hash), get_signed_url=True)
    encord_url_dict = {}
    for image in images or []:
        encord_url_dict[uuid.UUID(image["data_hash"])] = str(image["file_link"])
    if video is not None:
        encord_url_dict[data_hash] = str(video["file_link"])
    return encord_url_dict


@router.get("/{project_hash}/preview/{item}")
def display_preview(
    project_hash: uuid.UUID,
    preview_item: DataOrAnnotateItem = Depends(parse_data_or_annotate_item),
    engine: Engine = Depends(dep_engine),
):  # FIXME: type
    du_hash = preview_item.du_hash
    frame = preview_item.frame
    annotation_hash = preview_item.annotation_hash
    objects: List[dict] = []
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        query = select(
            ProjectDataUnitMetadata.data_uri,
            ProjectDataUnitMetadata.data_uri_is_video,
            ProjectDataUnitMetadata.objects,
            ProjectDataUnitMetadata.classifications,
            ProjectDataUnitMetadata.data_hash,
        ).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
            ProjectDataUnitMetadata.du_hash == du_hash,
            ProjectDataUnitMetadata.frame == frame,
        )
        result = sess.exec(query).fetchone()
        if result is None:
            raise ValueError("Missing project data unit metadata")
        uri, uri_is_video, objects, classifications, data_hash = result  # type: ignore
        frames_per_second_list = sess.exec(
            select(ProjectDataMetadata.frames_per_second).where(
                ProjectDataMetadata.project_hash == project_hash, ProjectDataMetadata.data_hash == data_hash
            )
        ).all()
        if len(frames_per_second_list) != 1:
            raise ValueError("Missing project data metadata")
        frames_per_second = frames_per_second_list[0]

        if annotation_hash is None:
            query_tags = (
                select(
                    ProjectTag,
                )
                .where(
                    ProjectTaggedDataUnit.project_hash == project_hash,
                    ProjectTaggedDataUnit.du_hash == du_hash,
                    ProjectTaggedDataUnit.frame == frame,
                )
                .join(ProjectTaggedDataUnit, ProjectTaggedDataUnit.tag_hash == ProjectTag.tag_hash)
            )
        else:
            query_tags = (
                select(
                    ProjectTag,
                )
                .where(
                    ProjectTaggedAnnotation.project_hash == project_hash,
                    ProjectTaggedAnnotation.du_hash == du_hash,
                    ProjectTaggedAnnotation.frame == frame,
                    ProjectTaggedAnnotation.annotation_hash == annotation_hash,
                )
                .join(ProjectTaggedAnnotation, ProjectTaggedAnnotation.tag_hash == ProjectTag.tag_hash)
            )
        result_tags = sess.exec(query_tags).fetchall()
    if annotation_hash is not None:
        objects = [obj for obj in objects if obj["objectHash"] == annotation_hash]
        if len(objects) == 0:
            raise ValueError("ObjectHash does not exist at that frame")
    if uri is not None:
        settings = get_settings()
        root_path = settings.SERVER_START_PATH.expanduser().resolve()
        url_path = url_to_file_path(uri, root_path)
        if url_path is not None:
            uri = f"/projects_v2/{project_hash}/files/{du_hash}/{frame}"
    elif project is not None and project.project_remote_ssh_key_path is not None:
        uri = _cache_encord_img_lookup(
            ssh_key_path=project.project_remote_ssh_key_path,
            project_hash=project_hash,
            data_hash=data_hash,
        )[du_hash]
    else:
        raise ValueError(f"Cannot resolve project url: {project_hash} / {du_hash}")

    timestamp = None
    if uri_is_video:
        if frames_per_second is None:
            raise ValueError("Video defined but missing valid frames_per_second")
        timestamp = (float(int(frame)) + 0.5) / float(frames_per_second)

    return {"url": uri, "timestamp": timestamp, "objects": objects, "tags": [tag.tag_hash for tag in result_tags]}


class ProjectItem(BaseModel):
    metrics: Dict[str, Optional[float]]
    objects: list[dict]
    classifications: list[dict]
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


@router.get("/{project_hash}/item/{data_item}/")
def project_item(
    project_hash: uuid.UUID, item_details: DataItem = Depends(parse_data_item), engine: Engine = Depends(dep_engine)
) -> ProjectItem:
    du_hash = item_details.du_hash
    frame = item_details.frame
    with Session(engine) as sess:
        project_ssh_key_path = sess.exec(
            select(Project.project_remote_ssh_key_path).where(Project.project_hash == project_hash)
        ).first()
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
        object_analytics = sess.exec(
            select(ProjectAnnotationAnalytics).where(
                ProjectAnnotationAnalytics.project_hash == project_hash,
                ProjectAnnotationAnalytics.du_hash == du_hash,
                ProjectAnnotationAnalytics.frame == frame,
            )
        ).fetchall()
    if du_meta is None or du_analytics is None:
        raise ValueError

    uri = du_meta.data_uri
    uri_is_video = du_meta.data_uri_is_video
    frames_per_second = data_meta.frames_per_second
    if uri is not None:
        settings = get_settings()
        root_path = settings.SERVER_START_PATH.expanduser().resolve()
        url_path = url_to_file_path(uri, root_path)
        if url_path is not None:
            uri = f"/projects_v2/{project_hash}/files/{du_hash}/{frame}"
    elif project_ssh_key_path is not None:
        uri = _cache_encord_img_lookup(
            ssh_key_path=project_ssh_key_path,
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

    return ProjectItem(
        metrics={k: getattr(du_analytics, k) for k in DataMetrics},
        objects=du_meta.objects,
        classifications=du_meta.classifications,
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
    )


class ListProjectPredictionResultEntry(BaseModel):
    name: str
    prediction_hash: uuid.UUID


class ListProjectPredictionResult(BaseModel):
    total: int
    results: List[ListProjectPredictionResultEntry]


@router.get("/{project_hash}/predictions")
def list_project_predictions(
    project_hash: uuid.UUID,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    order_by: Optional[Literal[""]] = None,
    engine: Engine = Depends(dep_engine),
) -> ListProjectPredictionResult:
    with Session(engine) as sess:
        predictions = sess.exec(
            select(ProjectPrediction).where(ProjectPrediction.project_hash == project_hash)
        ).fetchall()
    return ListProjectPredictionResult(
        total=len(predictions),
        results=[
            ListProjectPredictionResultEntry(name=prediction.name, prediction_hash=prediction.prediction_hash)
            for prediction in predictions
        ],
    )


@router.post("/{project_hash}/create/tag")
def create_data_tag(
    project_hash: uuid.UUID, tagged_items: List[str], tag_name: str, engine: Engine = Depends(dep_engine)
) -> None:
    tag_hash = uuid.uuid4()
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
        tags = [
            ProjectTaggedAnnotation(
                project_hash=project_hash,
                du_hash=du_hash,
                frame=frame,
                annotation_hash=annotation_hash,
                tag_hash=tag_hash,
            )
            for du_hash, frame, annotation_hash in annotations
        ]
        sess.add_all(tags)
        sess.commit()
