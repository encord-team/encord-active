import uuid
from typing import Dict, List, Optional, Tuple, Literal
from urllib.parse import quote

from fastapi import APIRouter
from sqlalchemy import func
from sqlalchemy.sql.operators import in_op
from sqlmodel import Session, select

from encord_active.db.enums import EnumDefinition, DataEnums, AnnotationEnums
from encord_active.db.metrics import AnnotationMetrics, DataMetrics, MetricDefinition
from encord_active.db.models import (
    Project,
    ProjectAnnotationAnalytics,
    ProjectDataAnalytics,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectPrediction,
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit, AnnotationType, ProjectEmbeddingReduction,
)
from encord_active.lib.common.data_utils import url_to_file_path
from encord_active.lib.encord.utils import get_encord_project
from encord_active.server.routers import project2_analysis, project2_prediction, project2_actions
from encord_active.server.routers.project2_engine import engine
from encord_active.server.settings import get_settings

router = APIRouter(
    prefix="/projects_v2",
    tags=["projects_v2"],
    # FIXME: dependencies=[Depends(verify_token)],
)
router.include_router(project2_analysis.router)
router.include_router(project2_prediction.router)
router.include_router(project2_actions.router)


@router.get("/")
def get_all_projects():
    with Session(engine) as sess:
        projects = sess.exec(select(Project.project_hash, Project.project_name, Project.project_description)).fetchall()
    return {
        project_hash: {
            "title": title,
            "description": description,
        }
        for project_hash, title, description in projects
    }


def _metric_summary(metrics: Dict[str, MetricDefinition]):
    return {
        metric_name: {
            "title": metric.title,
            "short_desc": metric.short_desc,
            "long_desc": metric.long_desc,
            "type": metric.type.value,
        }
        for metric_name, metric in metrics.items()
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
            select(func.count())
            .where(ProjectDataAnalytics.project_hash == project_hash)
            .group_by(ProjectDataAnalytics.du_hash)
        ).first()
        frame_count = sess.exec(
            select(func.count())
            .where(ProjectDataAnalytics.project_hash == project_hash)
            .group_by(ProjectDataAnalytics.du_hash)
        ).first()
        annotation_count = sess.exec(
            select(func.count())
            .where(ProjectAnnotationAnalytics.project_hash == project_hash)
        ).first()
        classification_count = sess.exec(
            select(func.count())
            .where(
                ProjectAnnotationAnalytics.project_hash == project_hash,
                ProjectAnnotationAnalytics.annotation_type == AnnotationType.CLASSIFICATION,
            )
        ).first()

    return {
        "name": project.project_name,
        "description": project.project_description,
        "ontology": project.project_ontology,
        "data": {
            "metrics": _metric_summary(DataMetrics),
            "enums": _enum_summary(DataEnums),
        },
        "annotations": {
            "metrics": _metric_summary(AnnotationMetrics),
            "enums": _enum_summary(AnnotationEnums),
        },
        "du_count": du_count,
        "frame_count": frame_count,
        "annotation_count": annotation_count,
        "classification_count": classification_count,
        # "global": {
        #    "metrics": _metric_summary(AnnotationMetrics | DataMetrics),
        #    "enums": _enum_summary(AnnotationEnums | DataEnums),
        # },
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
            relative_path = url_path.relative_to(root_path)
            uri = f"{settings.API_URL}/ea-static/{quote(relative_path.as_posix())}"
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
            raise ValueError(f"Video defined but missing valid frames_per_second")
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

