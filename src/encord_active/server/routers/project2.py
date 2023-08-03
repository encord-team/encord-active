import uuid
from typing import Dict, List, Literal, Optional, Set, Tuple, TypedDict

from cachetools import TTLCache, cached
from encord.objects import OntologyStructure
from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select
from starlette.responses import FileResponse

from encord_active.db.enums import AnnotationEnums, DataEnums, EnumDefinition
from encord_active.db.metrics import AnnotationMetrics, DataMetrics, MetricDefinition
from encord_active.db.models import (
    AnnotationType,
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
from encord_active.server.routers import (
    project2_actions,
    project2_analysis,
    project2_prediction,
)
from encord_active.server.routers.project2_engine import engine
from encord_active.server.routers.queries.metric_query import sql_count
from encord_active.server.settings import get_settings

router = APIRouter(
    prefix="/projects_v2",
    tags=["projects_v2"],
)
router.include_router(project2_analysis.router)
router.include_router(project2_prediction.router)
router.include_router(project2_actions.router)


class ProjectStats(TypedDict):
    dataUnits: int
    labels: int
    classes: int


class ProjectReturn(TypedDict):
    title: str
    description: str
    projectHash: uuid.UUID
    imageUrl: str
    imageUrlTimestamp: Optional[float]
    downloaded: bool
    sandbox: bool
    stats: Optional[ProjectStats]


def _get_first_image_with_polygons_url(project_hash: uuid.UUID) -> Optional[Tuple[str, float]]:
    with Session(engine) as sess:
        data_units = sess.exec(
            select(ProjectDataUnitMetadata).where(ProjectDataUnitMetadata.project_hash == project_hash).limit(100)
        ).fetchall()
        for data_unit in data_units:
            if not data_unit:
                break
            preview = display_preview(project_hash=project_hash, du_hash=data_unit.du_hash, frame=data_unit.frame)
            return preview["url"], preview["timestamp"]
    return None


@router.get("")
@cached(cache=TTLCache(maxsize=1, ttl=60 * 1))  # 1 minute TTL Cache
def get_all_projects() -> Dict[str, ProjectReturn]:
    # FIXME: revert back to this code!
    # with Session(engine) as sess:
    #  projects = sess.exec(select(Project.project_hash, Project.project_name, Project.project_description)).fetchall()
    # return {
    #     project_hash: {
    #         "title": title,
    #         "description": description,
    #     }
    #     for project_hash, title, description in projects
    # }
    sandbox_projects = {}
    for name, data in available_prebuilt_projects(get_settings().AVAILABLE_SANDBOX_PROJECTS).items():
        project_hash_uuid = uuid.UUID(data["hash"])
        sandbox_projects[str(project_hash_uuid)] = ProjectReturn(
            title=name,
            description="",
            projectHash=project_hash_uuid,
            stats=data["stats"],
            downloaded=False,
            imageUrl=f"/ea-sandbox-static/{data['image_filename']}",
            imageUrlTimestamp=None,
            sandbox=True,
        )

    with Session(engine) as sess:
        db_projects = sess.exec(
            select(Project.project_hash, Project.project_name, Project.project_description, Project.project_ontology)
        ).fetchall()
        data_count = sess.exec(
            select(ProjectDataMetadata.project_hash, sql_count()).group_by(ProjectDataMetadata.project_hash)
        ).fetchall()
        data_count_dict = {p: c for p, c in data_count}
        annotation_count = sess.exec(
            select(ProjectAnnotationAnalytics.project_hash, sql_count()).group_by(
                ProjectAnnotationAnalytics.project_hash
            )
        )
        annotation_count_dict = {p: c for p, c in annotation_count}
    projects = {}
    for project_hash, title, description, ontology in db_projects:
        if str(project_hash) in sandbox_projects:
            sandbox_projects[str(project_hash)]["downloaded"] = True
            # TODO: remove me when we can download sandbox projects from the UI
            sandbox_projects[str(project_hash)]["sandbox"] = False
            continue
        url_timestamp = _get_first_image_with_polygons_url(project_hash)
        url: Optional[str] = None
        timestamp: Optional[float] = None
        if url_timestamp is not None:
            url, timestamp = url_timestamp
        ontology = OntologyStructure.from_dict(ontology)
        label_classes = len(ontology.objects) + len(ontology.classifications)  # type: ignore
        projects[str(project_hash)] = ProjectReturn(
            title=title,
            description=description,
            projectHash=project_hash,
            stats=ProjectStats(
                dataUnits=data_count_dict.get(project_hash, 0),
                labels=annotation_count_dict.get(project_hash, 0),
                classes=label_classes,
            ),
            downloaded=True,
            imageUrl=url or "",
            imageUrlTimestamp=timestamp,
            sandbox=False,
        )
    return {**projects, **sandbox_projects}


def _metric_summary(metrics: Dict[str, MetricDefinition], fixme_exclude: Optional[Set[str]] = None):
    return {
        metric_name: {
            "title": metric.title,
            "short_desc": metric.short_desc,
            "long_desc": metric.long_desc,
            "type": metric.type.value,
        }
        for metric_name, metric in metrics.items()
        if fixme_exclude is None or metric_name not in fixme_exclude
    }


def _enum_summary(enums: Dict[str, EnumDefinition]):
    return {
        enum_name: {
            "title": enum.title,
            "values": enum.values,
            "type": enum.enum_type.value,
        }
        for enum_name, enum in enums.items()
    }


@router.get("/{project_hash}/summary")
def get_project_summary(project_hash: uuid.UUID):
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

    return {
        "name": project.project_name,
        "description": project.project_description,
        "ontology": project.project_ontology,
        "local_project": project.project_remote_ssh_key_path is None,
        "data": {
            "metrics": _metric_summary(DataMetrics),
            "enums": _enum_summary(DataEnums),
        },
        "annotations": {
            "metrics": _metric_summary(
                AnnotationMetrics,
                # Metrics only generated by WIP new computation engine. So exclude
                #  while only displaying migrated metrics
                # FIXME: remove this in the future when new computation engine is ready to land
                # fixme_exclude={
                #    "metric_width",
                #    "metric_height",
                #    "metric_area",
                #    "metric_aspect_ratio",
                #    "metric_brightness",
                #    "metric_contrast",
                #    "metric_sharpness",
                #    "metric_red",
                #    "metric_green",
                #    "metric_blue",
                # },
            ),
            "enums": _enum_summary(AnnotationEnums),
        },
        "du_count": du_count,
        "frame_count": frame_count,
        "annotation_count": annotation_count,
        "classification_count": classification_count,
        "global": {
            "metrics": _metric_summary(AnnotationMetrics | DataMetrics),
            "enums": _enum_summary(AnnotationEnums | DataEnums),
        },
        "tags": {tag.tag_hash: tag.name for tag in tags},
        "preview": {"du_hash": preview[0], "frame": preview[1]} if preview is not None else None,
    }


@router.get("/{project_hash}/reductions")
def list_supported_2d_embedding_reductions(project_hash: uuid.UUID):
    with Session(engine) as sess:
        r = sess.exec(select(ProjectEmbeddingReduction).where(ProjectEmbeddingReduction.project_hash == project_hash))
    return {
        "results": {
            e.reduction_hash: {
                "name": e.reduction_name,
                "description": e.reduction_description,
                "type": e.reduction_type.value,
            }
            for e in r
        }
    }


@router.get("/{project_hash}/files/{du_hash}/{frame}")
def display_raw_file(project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int):
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
                return FileResponse(url_path)
        raise HTTPException(status_code=404, detail="Project local file not found")


@router.get("/{project_hash}/preview/{du_hash}/{frame}/{object_hash}")
def display_preview(project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int, object_hash: Optional[str] = None):
    objects: List[dict] = []
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        query = select(
            ProjectDataUnitMetadata.data_uri,
            ProjectDataUnitMetadata.data_uri_is_video,
            ProjectDataUnitMetadata.objects,
            ProjectDataUnitMetadata.data_hash,
        ).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
            ProjectDataUnitMetadata.du_hash == du_hash,
            ProjectDataUnitMetadata.frame == frame,
        )
        result = sess.exec(query).fetchone()
        if result is None:
            raise ValueError("Missing project data unit metadata")
        uri, uri_is_video, objects, data_hash = result  # type: ignore
        data_metadata = sess.exec(
            select(ProjectDataMetadata.label_hash, ProjectDataMetadata.frames_per_second).where(
                ProjectDataMetadata.project_hash == project_hash, ProjectDataMetadata.data_hash == data_hash
            )
        ).first()
        if data_metadata is None:
            raise ValueError("Missing project data metadata")
        label_hash, frames_per_second = data_metadata
        if label_hash is None:
            raise ValueError("Missing label_hash")

        if object_hash is None:
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
                    ProjectTaggedAnnotation.object_hash == object_hash,
                )
                .join(ProjectTaggedAnnotation, ProjectTaggedAnnotation.tag_hash == ProjectTag.tag_hash)
            )
        result_tags = sess.exec(query_tags).fetchall()
    if object_hash is not None:
        objects = [obj for obj in objects if obj["objectHash"] == object_hash]
        if len(objects) == 0:
            raise ValueError("ObjectHash does not exist at that frame")
    if uri is not None:
        settings = get_settings()
        root_path = settings.SERVER_START_PATH.expanduser().resolve()
        url_path = url_to_file_path(uri, root_path)
        if url_path is not None:
            uri = f"/projects_v2/{project_hash}/files/{du_hash}/{frame}"
    elif project is not None and project.project_remote_ssh_key_path is not None:
        encord_project = get_encord_project(
            ssh_key_path=project.project_remote_ssh_key_path, project_hash=str(project_hash)
        )
        uri = encord_project.get_label_row(str(label_hash), get_signed_url=True,)["data_units"][
            str(du_hash)
        ]["data_link"]
    else:
        raise ValueError(f"Cannot resolve project url: {project_hash} / {label_hash} / {du_hash}")

    timestamp = None
    if uri_is_video:
        if frames_per_second is None:
            raise ValueError("Video defined but missing valid frames_per_second")
        timestamp = (float(int(frame)) + 0.5) / float(frames_per_second)

    return {"url": uri, "timestamp": timestamp, "objects": objects, "tags": [tag.tag_hash for tag in result_tags]}


@router.get("/{project_hash}/preview/{du_hash}/{frame}/")
def display_preview_data(project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int):
    return display_preview(project_hash, du_hash, frame)


@router.get("/{project_hash}/item/{du_hash}/{frame}/")
def item(project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int):
    with Session(engine) as sess:
        du_meta = sess.exec(
            select(ProjectDataUnitMetadata).where(
                ProjectDataUnitMetadata.project_hash == project_hash,
                ProjectDataUnitMetadata.du_hash == du_hash,
                ProjectDataUnitMetadata.frame == frame,
            )
        ).first()
        if du_meta is None:
            raise ValueError("Invalid request")
        data_meta = sess.exec(
            select(ProjectDataMetadata).where(
                ProjectDataMetadata.project_hash == project_hash, ProjectDataMetadata.data_hash == du_meta.data_hash
            )
        ).first()
        if data_meta is None:
            raise ValueError("Invalid request")
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
    obj_metrics = {o.object_hash: o for o in object_analytics or []}

    return {
        "metrics": {k: getattr(du_analytics, k) for k in DataMetrics},
        "objects": [
            {
                **obj,
                "metrics": {k: getattr(obj_metrics[obj["objectHash"]], k) for k in AnnotationMetrics},
                "tags": [],
            }
            for obj in du_meta.objects
        ],
        "dataset_title": data_meta.dataset_title,
        "dataset_hash": data_meta.dataset_hash,
        "data_title": data_meta.data_title,
        "data_hash": data_meta.data_hash,
        "label_hash": data_meta.label_hash,
        "num_frames": data_meta.num_frames,
        "frames_per_second": data_meta.frames_per_second,
        "data_type": data_meta.data_type,
    }


@router.get("/{project_hash}/predictions")
def list_project_predictions(
    project_hash: uuid.UUID,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    order_by: Optional[Literal[""]] = None,
):
    with Session(engine) as sess:
        predictions = sess.exec(
            select(ProjectPrediction).where(ProjectPrediction.project_hash == project_hash)
        ).fetchall()
    return {
        "total": len(predictions),
        "results": [
            {"name": prediction.name, "prediction_hash": prediction.prediction_hash} for prediction in predictions
        ],
    }


@router.post("/{project_hash}/create/tag/data")
def create_data_tag(project_hash: uuid.UUID, data: List[Tuple[uuid.UUID, int]], name: str):
    tag_hash = uuid.uuid4()
    with Session(engine) as sess:
        sess.add(ProjectTag(tag_hash=tag_hash, project_hash=project_hash, name=name))
        tags = [
            ProjectTaggedDataUnit(project_hash=project_hash, du_hash=du_hash, frame=frame, tag_hash=tag_hash)
            for du_hash, frame in data
        ]
        sess.add_all(tags)
        sess.commit()


@router.post("/{project_hash}/create/tag/annotation")
def create_annotation_tag(project_hash: uuid.UUID, annotations: List[Tuple[uuid.UUID, int, str]], name: str):
    tag_hash = uuid.uuid4()
    with Session(engine) as sess:
        sess.add(ProjectTag(tag_hash=tag_hash, project_hash=project_hash, name=name))
        tags = [
            ProjectTaggedAnnotation(
                project_hash=project_hash, du_hash=du_hash, frame=frame, object_hash=object_hash, tag_hash=tag_hash
            )
            for du_hash, frame, object_hash in annotations
        ]
        sess.add_all(tags)
        sess.commit()
