import json
import uuid
from enum import Enum
from typing import Optional, Dict, Tuple, Union, List, Type, Literal
from urllib.parse import quote

import numpy as np

from fastapi import APIRouter
from sqlalchemy import func, bindparam, text
from sqlalchemy.types import Float
from sqlalchemy.sql.operators import is_not, in_op, not_between_op, between_op
from sqlmodel import Session, select
from sqlmodel.sql.sqltypes import GUID

from encord_active.db.metrics import DataMetrics, ClassificationMetrics, ObjectMetrics, MetricDefinition, MetricType
from encord_active.db.models import get_engine, Project, ProjectDataUnitMetadata, ProjectTaggedDataUnit, \
    ProjectTaggedObject, ProjectTag, ProjectDataAnalytics, ProjectObjectAnalytics, ProjectDataMetadata, \
    ProjectClassificationAnalytics, ProjectPrediction, ProjectPredictionObjectResults
from encord_active.lib.common.data_utils import url_to_file_path
from encord_active.lib.encord.utils import get_encord_project
from encord_active.server.settings import get_settings

router = APIRouter(
    prefix="/projects_v2",
    tags=["projects_v2"],
    # FIXME: dependencies=[Depends(verify_token)],
)

engine_path = get_settings().SERVER_START_PATH / "encord-active.sqlite"
engine = get_engine(engine_path, concurrent=True)

MODERATE_IQR_SCALE = 1.5
SEVERE_IQR_SCALE = 2.5


class AnalysisDomain(Enum):
    Object = "object"
    Data = "data"
    Classification = "classification"


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


def _metric_summary(metrics: Dict[str, MetricDefinition]):
    return {
        metric_name: {
            "title": metric.title,
            "short_desc": metric.short_desc,
            "long_desc": metric.long_desc,
            "type": metric.type.name,
        }
        for metric_name, metric in metrics.items()
    }


@router.get("/get/{project_hash}/summary")
def get_project_summary(project_hash: uuid.UUID):
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError()

        tags = sess.exec(select(ProjectTag).where(ProjectTag.project_hash == project_hash)).fetchall()

        preview = sess.exec(select(ProjectObjectAnalytics.du_hash, ProjectObjectAnalytics.frame)
                            .where(ProjectObjectAnalytics.project_hash == project_hash).limit(1)).first()
        if preview is None:
            preview = sess.exec(select(ProjectDataUnitMetadata.du_hash, ProjectDataUnitMetadata.frame)
                                .where(ProjectDataUnitMetadata.project_hash == project_hash).limit(1)).first()

    return {
        "name": project.project_name,
        "description": project.project_description,
        "ontology": project.project_ontology,
        "data": {
            "metrics": _metric_summary(DataMetrics),
            "enums": {},
        },
        "objects": {
            "metrics": _metric_summary(ObjectMetrics),
            "enums": {
                "feature_hash": None,
            },
        },
        "classifications": {
            "metrics": _metric_summary(ClassificationMetrics),
            "enums": {
                "feature_hash": None,
            },
        },
        "tags": {
            tag.tag_hash: tag.name
            for tag in tags
        },
        "preview": {
            "du_hash": preview[0],
            "frame": preview[1]
        } if preview is not None else None
    }


@router.get("/get/{project_hash}/preview/{du_hash}/{frame}/{object_hash}")
def display_preview(project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int, object_hash: Optional[str] = None):
    objects = []
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
            ProjectDataUnitMetadata.frame == frame
        )
        result = sess.exec(query).fetchone()
        if result is None:
            raise ValueError("Missing project data unit metadata")
        uri, uri_is_video, objects, data_hash = result
        label_hash, frames_per_second = sess.exec(
            select(ProjectDataMetadata.label_hash, ProjectDataMetadata.frames_per_second)
            .where(ProjectDataMetadata.project_hash == project_hash,
                   ProjectDataMetadata.data_hash == data_hash)
        ).first()
        if label_hash is None:
            raise ValueError("Missing label_hash")

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
    if object_hash is not None:
        objects = [
            obj
            for obj in objects
            if obj["objectHash"] == object_hash
        ]
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
            ssh_key_path=project.project_remote_ssh_key_path,
            project_hash=str(project_hash)
        )
        uri = encord_project.get_label_row(
            label_hash, get_signed_url=True,
        )["data_units"][du_hash]["data_link"]
    else:
        raise ValueError(f"Cannot resolve project url")

    timestamp = None
    if uri_is_video:
        timestamp = (float(int(frame)) + 0.5) / frames_per_second

    return {
        "url": uri,
        "timestamp": timestamp,
        "objects": objects,
        "tags": [
            tag.tag_hash
            for tag in result_tags
        ]
    }


@router.get("/get/{project_hash}/preview/{du_hash}/{frame}/")
def display_preview_data(project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int):
    return display_preview(project_hash, du_hash, frame)


@router.get("/get/{project_hash}/item/{du_hash}/{frame}/")
def item(project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int):
    with Session(engine) as sess:
        du_meta = sess.exec(select(ProjectDataUnitMetadata).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
            ProjectDataUnitMetadata.du_hash == du_hash,
            ProjectDataUnitMetadata.frame == frame
        )).first()
        du_analytics = sess.exec(
            select(
                ProjectDataAnalytics
            ).where(
                ProjectDataAnalytics.project_hash == project_hash,
                ProjectDataAnalytics.du_hash == du_hash,
                ProjectDataAnalytics.frame == frame,
            )
        ).first()
        object_analytics = sess.exec(
            select(
                ProjectObjectAnalytics
            ).where(
                ProjectObjectAnalytics.project_hash == project_hash,
                ProjectObjectAnalytics.du_hash == du_hash,
                ProjectObjectAnalytics.frame == frame,
            )
        ).fetchall()
        classification_analytics = sess.exec(
            select(
                ProjectClassificationAnalytics
            ).where(
                ProjectClassificationAnalytics.project_hash == project_hash,
                ProjectClassificationAnalytics.du_hash == du_hash,
                ProjectClassificationAnalytics.frame == frame,
            )
        ).fetchall()
    if du_meta is None or du_analytics is None:
        raise ValueError
    obj_metrics = {
        o.object_hash: o
        for o in object_analytics or []
    }
    classify_metrics = {
        c.classification_hash: c
        for c in classification_analytics or []
    }

    return {
        "metrics": {
            k: getattr(du_analytics, k)
            for k in DataMetrics
        },
        "objects": [
            {
                **obj,
                "metrics": {
                    k: getattr(obj_metrics[obj["objectHash"]], k)
                    for k in ObjectMetrics
                },
                "tags": [],
            }
            for obj in du_meta.objects
        ],
        "classifications": [
            {
                **classify,
                "metrics": {
                    k: getattr(classify_metrics[classify["classificationHash"]], k)
                    for k in ClassificationMetrics
                },
                "tags": [],
            }
            for classify in du_meta.classifications
        ]
    }


def _get_metric_domain(domain: AnalysisDomain) -> Tuple[
    Union[Type[ProjectDataAnalytics], Type[ProjectObjectAnalytics], Type[ProjectClassificationAnalytics]],
    Dict[str, MetricDefinition],
    Optional[str],
    Dict[str, None]
]:
    if domain == AnalysisDomain.Data:
        return ProjectDataAnalytics, DataMetrics, None, {}
    elif domain == AnalysisDomain.Object:
        return ProjectObjectAnalytics, ObjectMetrics, "object_hash", {"feature_hash": None}
    elif domain == AnalysisDomain.Classification:
        return ProjectClassificationAnalytics, ClassificationMetrics, "classification_hash", {"feature_hash": None}
    else:
        raise ValueError(f"Bad domain: {domain}")


def _where_metric_not_null(cls, metric_name: str, metrics: Dict[str, MetricDefinition]):
    metric = metrics[metric_name]
    if metric.virtual is not None:
        metric_name = metric.virtual.src
    return is_not(getattr(cls, metric_name), None)


def _get_metric(cls, metric_name: str, metrics: Dict[str, MetricDefinition]):
    metric = metrics[metric_name]
    if metric.virtual is not None:
        return metric.virtual.map(getattr(cls, metric.virtual.src))
    return getattr(cls, metric_name)


def _load_metric(
        sess: Session,
        project_hash: uuid.UUID,
        metric_name: str,
        metric: MetricDefinition,
        cls: Type[Union[ProjectDataAnalytics, ProjectObjectAnalytics, ProjectClassificationAnalytics]],
        iqr_only: bool = False
) -> Optional[dict]:
    if metric.virtual is not None:
        metric_attr = getattr(cls, metric.virtual.src)
    else:
        metric_attr = getattr(cls, metric_name)
    where = [
        cls.project_hash == project_hash,
        is_not(metric_attr, None)
    ]
    count: int = sess.exec(select(func.count()).where(*where)).first()
    if count == 0:
        return None

    median_offset = count // 2
    q1_offset = count // 4
    q3_offset = (count * 3) // 4

    # Calculate base statistics
    if iqr_only:
        metric_min = 0
        metric_max = 0
        median = 0
    else:
        metric_min = sess.exec(select(func.min(metric_attr)).where(*where)).first()
        metric_max = sess.exec(select(func.max(metric_attr)).where(*where)).first()
        median = sess.exec(
            select(metric_attr).where(*where).order_by(metric_attr).offset(median_offset).limit(1)).first()
    q1 = sess.exec(select(metric_attr).where(*where).order_by(metric_attr).offset(q1_offset).limit(1)).first()
    q3 = sess.exec(select(metric_attr).where(*where).order_by(metric_attr).offset(q3_offset).limit(1)).first()

    # Calculate count of moderate & severe outliers
    if iqr_only:
        severe_count = 0
        moderate_count = 0
    else:
        iqr = q3 - q1
        moderate_lb, moderate_ub = q1 - MODERATE_IQR_SCALE * iqr, q3 + MODERATE_IQR_SCALE * iqr
        severe_lb, severe_ub = q1 - SEVERE_IQR_SCALE * iqr, q3 + SEVERE_IQR_SCALE * iqr
        severe_count = sess.exec(select(func.count()).where(
            *where,
            not_between_op(metric_attr, severe_lb, severe_ub)
        )).first()
        moderate_count = sess.exec(select(func.count()).where(
            *where,
            not_between_op(metric_attr, moderate_lb, moderate_ub),
            between_op(metric_attr, severe_lb, severe_ub)
        )).first()

    # Apply virtual metric transformations
    if metric.virtual is not None:
        metric_min = metric.virtual.map(metric_min)
        q1 = metric.virtual.map(q1)
        median = metric.virtual.map(median)
        q3 = metric.virtual.map(q3)
        metric_max = metric.virtual.map(metric_max)
        if metric.virtual.flip_ord:
            metric_min, metric_max = metric_max, metric_min
            q1, q3 = q3, q1

    return {
        "min": metric_min,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": metric_max,
        "count": count,
        "moderate": moderate_count,
        "severe": severe_count,
    }


@router.get("/get/{project_hash}/analysis/{domain}/summary")
def metric_summary(
        project_hash: uuid.UUID,
        domain: AnalysisDomain
):
    domain_ty, domain_metrics, extra_key, domain_enums, *_ = _get_metric_domain(domain)
    with Session(engine) as sess:
        count: int = sess.exec(select(func.count()).where(
            domain_ty.project_hash == project_hash
        )).first()
    metrics = {
        metric_name: _load_metric(sess, project_hash, metric_name, metric, domain_ty)
        for metric_name, metric in domain_metrics.items()
    }
    return {
        "count": count,
        "metrics": {
            k: v
            for k, v in metrics.items()
            if v is not None
        },
        "enums": domain_enums,
    }


@router.get("/get/{project_hash}/analysis/{domain}/search")
def metric_search(
        project_hash: uuid.UUID,
        domain: AnalysisDomain,
        metric_filters: Optional[str] = None,
        metric_outliers: Optional[str] = None,
        enum_filters: Optional[str] = None,
        order_by: Optional[str] = None,
        desc: bool = False
):
    metric_filters: Optional[Dict[str, Tuple[Union[int, float], Union[int, float]]]] = None if metric_filters is None \
        else json.loads(metric_filters)
    metric_outliers: Optional[Dict[str, Literal["warning", "severe"]]] = None if metric_outliers is None \
        else json.loads(metric_outliers)
    enum_filters: Optional[Dict[str, List[str]]] = None if enum_filters is None else json.loads(enum_filters)
    domain_ty, domain_metrics, domain_grouping, *_ = _get_metric_domain(domain)

    # Add metric filtering.
    query_filters = []
    if metric_filters is not None:
        for metric_name, (range_start, range_end) in metric_filters.items():
            metric_meta = domain_metrics[metric_name]
            metric_filter = getattr(domain_ty, metric_name)
            if metric_meta.virtual is not None:
                range_start = metric_meta.virtual.map(range_start)
                range_end = metric_meta.virtual.map(range_end)
                if metric_meta.virtual.flip_ord:
                    range_start, range_end = range_end, range_start
                metric_filter = getattr(domain_ty, metric_meta.virtual.src)
            if range_start == range_end:
                query_filters.append(
                    metric_filter == range_start
                )
            else:
                query_filters.append(
                    metric_filter >= range_start
                )
                query_filters.append(
                    metric_filter <= range_end
                )

    if enum_filters is not None:
        for enum_filter_name, enum_filter_list in enum_filters.items():
            if enum_filter_list not in ["feature_hash"]:
                raise ValueError(f"Unsupported enum filter: {enum_filter_name}")
            enum_filter_col = getattr(domain_ty, enum_filter_name)
            query_filters.append(
                in_op(enum_filter_col, enum_filter_list)
            )

    with Session(engine) as sess:
        if metric_outliers is not None:
            for metric, outlier_type in metric_outliers:
                metric_info = domain_metrics[metric]
                if metric_info.virtual is not None:
                    raw_metric = metric_info.virtual.src
                else:
                    raw_metric = metric
                summary = _load_metric(
                    sess, project_hash, raw_metric, domain_metrics[raw_metric], domain_ty,
                    iqr_only=True
                )
                metric_attr = getattr(domain_ty, raw_metric)
                q1 = summary["q1"]
                q3 = summary["q3"]
                iqr = q3 - q1
                moderate_lb, moderate_ub = q1 - MODERATE_IQR_SCALE * iqr, q3 + MODERATE_IQR_SCALE * iqr
                severe_lb, severe_ub = q1 - SEVERE_IQR_SCALE * iqr, q3 + SEVERE_IQR_SCALE * iqr
                if outlier_type == "severe":
                    query_filters.append(not_between_op(metric_attr, severe_lb, severe_ub))
                else:
                    query_filters.append(not_between_op(metric_attr, moderate_lb, moderate_ub))
                    query_filters.append(between_op(metric_attr, severe_lb, severe_ub))

        search_query = select(
            domain_ty.du_hash,
            domain_ty.frame,
            None if domain_grouping is None else getattr(domain_ty, domain_grouping)
        ).where(
            domain_ty.project_hash == project_hash,
            *query_filters
        )
        if order_by is not None:
            order_by_field = getattr(domain_ty, order_by)
            if desc:
                order_by_field = order_by_field.desc()
            search_query = search_query.order_by(
                order_by_field
            )

        search_query = search_query.limit(
            # 10_000 max results in a search query (+1 to detect truncation).
            limit=10_001
        )
        search_results = sess.exec(search_query).fetchall()
    truncated = len(search_results) == 10_001

    return {
        "truncated": truncated,
        "results": [
            {"du_hash": du_hash, "frame": frame, domain_grouping: group_hash}
            for du_hash, frame, group_hash in search_results
        ] if domain_grouping is not None else [
            {"du_hash": du_hash, "frame": frame}
            for du_hash, frame, group_hash in search_results
        ]
    }


@router.get("/get/{project_hash}/analysis/{domain}/scatter")
def scatter_2d_data_metric(project_hash: uuid.UUID, domain: AnalysisDomain, x_metric: str, y_metric: str):
    domain_ty, domain_metrics, *_ = _get_metric_domain(domain)
    with Session(engine) as sess:
        scatter_query = select(
            domain_ty.du_hash,
            domain_ty.frame,
            _get_metric(domain_ty, x_metric, domain_metrics),
            _get_metric(domain_ty, y_metric, domain_metrics),
        ).where(
            domain_ty.project_hash == project_hash,
            _where_metric_not_null(domain_ty, x_metric, domain_metrics),
            _where_metric_not_null(domain_ty, y_metric, domain_metrics),
        )
        scatter_results = sess.exec(scatter_query).fetchall()
    samples = [
        {"x": x, "y": y, "n": 1, "du_hash": du_hash, "frame": frame}
        for du_hash, frame, x, y in scatter_results
    ]

    # Derive linear regression
    x = np.array(list(res[2] for res in scatter_results), dtype=np.float)
    y = np.array(list(res[3] for res in scatter_results), dtype=np.float)
    a = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(a, y, rcond=None)[0]

    return {
        "sampling": 1.0,
        "samples": samples,
        "m": m,
        "c": c,
    }


@router.get("/get/{project_hash}/analysis/{domain}/dist")
def get_metric_distribution(project_hash: uuid.UUID, domain: AnalysisDomain, group: str):
    domain_ty, domain_metrics, object_key, domain_enums, *_ = _get_metric_domain(domain)
    if group in domain_metrics:
        metric = domain_metrics[group]
        if metric.virtual is not None:
            filter_attr = getattr(domain_ty, metric.virtual.src)
            metric_attr = metric.virtual.map(filter_attr)
        else:
            metric_attr = getattr(domain_ty, group)
            filter_attr = metric_attr
        if metric.type == MetricType.NORMAL:
            group_by_attr = func.floor(metric_attr * 100.0) / 100.0
        else:
            group_by_attr = metric_attr
    elif group in domain_enums:
        group_by_attr = getattr(domain_ty, group)
        filter_attr = group_by_attr
    else:
        raise ValueError(f"{group} is not a valid distribution key")

    with Session(engine) as sess:
        grouping_query = select(
            group_by_attr,
            func.count()
        ).where(
            domain_ty.project_hash == project_hash,
            is_not(filter_attr, None)
        ).group_by(group_by_attr)
        grouping_results = sess.exec(
            grouping_query
        ).fetchall()

    return {
        "results": [
            {
                "group": grouping,
                "count": count,
            }
            for grouping, count in grouping_results
        ]
    }


@router.get("/get/{project_hash}/predictions/list")
def list_project_predictions(project_hash: uuid.UUID, offset: Optional[int] = None, limit: Optional[int] = None):
    with Session(engine) as sess:
        predictions = sess.exec(select(ProjectPrediction)
                                .where(ProjectPrediction.project_hash == project_hash)).fetchall()
    return {
        "total": len(predictions),
        "results": [
            {
                "name": prediction.name,
                "prediction_hash": prediction.prediction_hash
            }
            for prediction in predictions
        ]
    }


@router.get("/get/{project_hash}/predictions/get/{prediction_hash}/summary")
def get_project_prediction_summary(project_hash: uuid.UUID, prediction_hash: uuid.UUID, iou: float):
    guid = GUID()
    with Session(engine) as sess:
        sql = text(
            """
            SELECT feature_hash, coalesce(ap, 0.0) as ap, coalesce(ar, 0.0) as ar FROM (
                SELECT FG.feature_hash, (
                    SELECT sum(
                        coalesce(cast(ap_positives as real) / ap_total, 0.0)
                    ) / (
                        count(*) + (
                            SELECT COUNT(*) FROM active_project_prediction_unmatched UM
                            WHERE UM.prediction_hash = FG.prediction_hash
                            AND UM.feature_hash == FG.feature_hash
                        )
                    ) as ap FROM (
                        SELECT
                            row_number() OVER (
                                ORDER BY AP.confidence DESC ROWS UNBOUNDED PRECEDING
                            ) AS ap_total,
                            sum(cast(
                                (
                                    AP.iou >= :iou and
                                    coalesce(AP.feature_hash = AP.match_feature_hash, false) and
                                    AP.match_duplicate_iou < :iou
                                ) as integer
                            )) OVER (
                                ORDER BY AP.confidence DESC ROWS UNBOUNDED PRECEDING
                            ) AS ap_positives
                        FROM active_project_prediction_object AP
                        WHERE AP.prediction_hash = FG.prediction_hash
                        AND AP.feature_hash == FG.feature_hash
                        ORDER BY AP.confidence DESC
                    )
                ) as ap,
                (
                    SELECT sum(
                        coalesce(cast(ar_positives as real) / ar_total, 0.0)
                    ) / (
                        count(*) + (
                            SELECT COUNT(*) FROM active_project_prediction_unmatched UM
                            WHERE UM.prediction_hash = FG.prediction_hash
                            AND UM.feature_hash == FG.feature_hash
                        )
                    ) FROM (
                        SELECT
                            row_number() OVER (
                                ORDER BY AR.iou DESC ROWS UNBOUNDED PRECEDING
                            ) AS ar_total,
                            sum(cast(
                                (
                                    AR.iou >= :iou and
                                    coalesce(AR.feature_hash = AR.match_feature_hash, false) and
                                    AR.match_duplicate_iou < :iou
                                ) as integer
                            )) OVER (
                                ORDER BY AR.iou DESC ROWS UNBOUNDED PRECEDING
                            ) AS ar_positives
                        FROM active_project_prediction_object AR
                        WHERE AR.prediction_hash = FG.prediction_hash
                        AND AR.feature_hash = FG.feature_hash
                        ORDER BY AR.iou
                    )
                ) as ar
                FROM active_project_prediction_object FG, active_project_prediction FP
                WHERE FG.prediction_hash = FP.prediction_hash
                AND FP.project_hash = :project_hash
                AND FP.prediction_hash = :prediction_hash
                GROUP BY FG.feature_hash
            )
            """
        ).bindparams(bindparam("prediction_hash", GUID), bindparam("project_hash", GUID), bindparam("iou", Float))
        precision_recall = sess.execute(
            sql,
            params={
                "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                "project_hash": guid.process_bind_param(project_hash, engine.dialect),
                "iou": iou
            },
        ).fetchall()

        # FIXME: sqlite does not have corr() function, so have to implement manually.
        sql = text(
            """
            SELECT
                avg(
                    ((
                        iou >= :iou and
                        coalesce(feature_hash = match_feature_hash, false) and
                        match_duplicate_iou < :iou
                    ) - e_score) * (metric - e_metric)) / (std_score * std_metric)
            FROM active_project_prediction_object, (
                SELECT 
                    avg(
                        iou >= :iou and
                        coalesce(feature_hash = match_feature_hash, false) and
                        match_duplicate_iou < :iou
                    ) as e_score,
                    stdev(
                        iou >= :iou and
                        coalesce(feature_hash = match_feature_hash, false) and
                        match_duplicate_iou < :iou
                    ) as std_score
                FROM active_project_prediction_object
            )
            """
        )

        # Precision recall curves
        prs = {}
        for pr_summary in precision_recall:
            feature_hash = pr_summary["feature_hash"]
            sql = text(
                """
                SELECT
                    cast(tp_count as real) / max(tp_and_fp_count) as p,
                    cast(tp_count as real) / (
                        (
                            SELECT COUNT(*) FROM active_project_prediction_object CTP
                            WHERE CTP.prediction_hash = :prediction_hash
                            AND CTP.feature_hash = :prediction_hash
                            AND CTP.iou >= :iou and
                            coalesce(CTP.feature_hash = CTP.match_feature_hash, false) and
                            CTP.match_duplicate_iou < :iou
                        ) + (
                            SELECT COUNT(*) FROM active_project_prediction_unmatched UM
                            WHERE UM.prediction_hash = :prediction_hash
                            AND UM.feature_hash == :feature_hash
                        )
                    )as r
                FROM (
                    SELECT
                        row_number() OVER (
                            ORDER BY PR.confidence DESC ROWS UNBOUNDED PRECEDING
                        ) AS tp_and_fp_count,
                        sum(cast(
                            (
                                PR.iou >= :iou and
                                coalesce(PR.feature_hash = PR.match_feature_hash, false) and
                                PR.match_duplicate_iou < :iou
                            ) as integer
                        )) OVER (
                            ORDER BY PR.confidence DESC ROWS UNBOUNDED PRECEDING
                        ) AS tp_count
                    FROM active_project_prediction_object PR, active_project_prediction FP
                    WHERE PR.prediction_hash = FP.prediction_hash
                    AND FP.project_hash = :project_hash
                    AND FP.prediction_hash = :prediction_hash
                    AND PR.feature_hash = feature_hash
                    ORDER BY confidence DESC
                ) GROUP BY tp_count
                """
            ).bindparams(bindparam("prediction_hash", GUID), bindparam("project_hash", GUID), bindparam("iou", Float))
            pr_result = sess.execute(
                sql,
                params={
                    "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                    "project_hash": guid.process_bind_param(project_hash, engine.dialect),
                    "iou": iou,
                    "feature_hash": feature_hash
                },
            ).fetchall()
            prs[feature_hash] = pr_result
    return {
        "mAP": sum(pr["ap"] for pr in precision_recall) / max(len(precision_recall), 1),
        "mAR": sum(pr["ar"] for pr in precision_recall) / max(len(precision_recall), 1),
        "precisions": {
            pr["feature_hash"]: pr["ap"]
            for pr in precision_recall
        },
        "recalls": {
            pr["feature_hash"]: pr["ar"]
            for pr in precision_recall
        },
        "correlation": {},
        "importance": {},
        "prs": prs
    }


@router.get("/get/{project_hash}/predictions/get/{prediction_hash}/precision")
def get_project_prediction_importance(project_hash: uuid.UUID, prediction_hash: uuid.UUID, metric: str):
    pass


@router.get("/get/{project_hash}/predictions/get/{prediction_hash}/correlation")
def get_project_prediction_correlation(project_hash: uuid.UUID, prediction_hash: uuid.UUID, metric: str):
    pass
