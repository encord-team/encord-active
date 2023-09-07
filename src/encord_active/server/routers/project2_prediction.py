import math
import uuid
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sklearn import metrics as sklearn_metrics
from sklearn.feature_selection import (
    mutual_info_regression as sklearn_mutual_info_regression,
)
from sqlalchemy import Float, Integer, bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.sql.functions import coalesce
from sqlalchemy.sql.operators import is_not
from sqlmodel import Session, select
from sqlmodel.sql.sqltypes import GUID

from encord_active.db.enums import AnnotationType
from encord_active.db.metrics import AnnotationMetrics
from encord_active.db.models import (
    ProjectAnnotationAnalytics,
    ProjectPrediction,
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsFalseNegatives,
    ProjectPredictionDataUnitMetadata,
    ProjectPredictionDataMetadata,
)
from encord_active.db.util.char8 import Char8
from encord_active.server.dependencies import dep_engine, DataItem, parse_data_item
from encord_active.server.routers import project2_prediction_analysis
from encord_active.server.routers.queries import metric_query, search_query
from encord_active.server.routers.queries.domain_query import (
    TABLES_ANNOTATION,
    TABLES_PREDICTION_TP_FP,
)
from encord_active.server.routers.queries.metric_query import (
    literal_bucket_depends,
    sql_count,
    sql_max,
    sql_min,
    sql_sum,
)
from encord_active.server.routers.queries.search_query import (
    SearchFiltersFastAPIDepends,
)


def dep_check_project_prediction_hash_match(
    project_hash: uuid.UUID, prediction_hash: uuid.UUID, engine: Engine = Depends(dep_engine)
):
    with Session(engine) as sess:
        exists = (
            sess.exec(
                select(1).where(
                    ProjectPrediction.prediction_hash == prediction_hash, ProjectPrediction.project_hash == project_hash
                )
            ).first()
            is not None
        )
        if not exists:
            raise HTTPException(status_code=404, detail="The prediction hash is not associated with this project")


router = APIRouter(
    prefix="/{project_hash}/predictions/{prediction_hash}",
    dependencies=[Depends(dep_check_project_prediction_hash_match)],
)
router.include_router(project2_prediction_analysis.router)


def remove_nan_inf(v: float, inf: float = 100_000.0) -> float:
    if math.isnan(v):
        return 0.0
    elif math.isinf(v):
        return inf if v > 0.0 else -inf
    else:
        return v


# FIXME: verify project_hash <=> prediction hash is allowed to access

# FIXME: for group by filter_hash queries, what is the correct behaviour r.e case where no value exists for that feature
#  hash


class PredictionSummaryFeatureResult(BaseModel):
    ap: float
    ar: float
    p: float
    r: float
    f1: float
    tp: int
    fp: int
    fn: int


class PredictionPRPoint(BaseModel):
    p: float
    r: float


class PredictionSummaryResult(BaseModel):
    # Classification summary
    classification_only: bool
    classification_tTN: Optional[int]
    classification_accuracy: Optional[float]
    # Common summary
    num_frames: int
    mAP: float
    mAR: float
    mP: float
    mR: float
    mF1: float
    tTP: int
    tFP: int
    tFN: int
    # Feature properties
    feature_properties: Dict[str, PredictionSummaryFeatureResult]
    prs: Dict[str, List[PredictionPRPoint]]
    # Metric properties
    correlation: Dict[str, float]
    importance: Dict[str, float]


#


@router.get("/summary")
def get_project_prediction_summary(
    prediction_hash: uuid.UUID,
    iou: float,
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine),
) -> PredictionSummaryResult:
    # FIXME: this command will return the wrong answers when filters are applied!!!
    tp_fp_where = search_query.search_filters(
        tables=TABLES_PREDICTION_TP_FP,
        base=ProjectPredictionAnalytics,
        search=filters,
        project_filters={
            "prediction_hash": [prediction_hash],
            # FIXME: project_hash
        },
    )
    # FIXME: fn_where needs separate set of queries as this is implemented as a side table join.
    # FIXME: performance improvements are possible
    guid = GUID()
    char8 = Char8()
    with Session(engine) as sess:
        range_annotation_types = sess.exec(
            select(  # type: ignore
                sql_min(ProjectPredictionAnalytics.annotation_type),
                sql_max(ProjectPredictionAnalytics.annotation_type),
            ).where(ProjectPredictionAnalytics.prediction_hash == prediction_hash)
        ).first()
        is_classification = False
        if range_annotation_types is not None:
            annotate_ty_min, annotate_ty_max = range_annotation_types
            is_classification = annotate_ty_max == annotate_ty_min and annotate_ty_min == AnnotationType.CLASSIFICATION

        # FIXME: this is only used by classification only predictions.
        # FIXME: this should be separated and considered in the future.
        num_frames: int = (
            sess.exec(
                select(sql_count()).select_from(
                    select(
                        ProjectPredictionAnalyticsFalseNegatives.du_hash,
                        ProjectPredictionAnalyticsFalseNegatives.frame,
                    )
                    .where(ProjectPredictionAnalyticsFalseNegatives.prediction_hash == prediction_hash)
                    .group_by(
                        ProjectPredictionAnalyticsFalseNegatives.du_hash,
                        ProjectPredictionAnalyticsFalseNegatives.frame,
                    )
                )
            ).first()
            or 0
        )

        # Select summary count of TP / FP / FN
        false_negative_counts_raw = sess.exec(
            select(
                ProjectPredictionAnalyticsFalseNegatives.feature_hash,
                sql_count(),
            )
            .where(
                ProjectPredictionAnalyticsFalseNegatives.prediction_hash == prediction_hash,
                ProjectPredictionAnalyticsFalseNegatives.iou_threshold < iou,
            )
            .group_by(
                ProjectPredictionAnalyticsFalseNegatives.feature_hash,
            )
        ).fetchall()
        false_negative_all_feature_hashes = sess.exec(
            select(
                ProjectPredictionAnalyticsFalseNegatives.feature_hash,
            )
            .where(
                ProjectPredictionAnalyticsFalseNegatives.prediction_hash == prediction_hash,
            )
            .group_by(
                ProjectPredictionAnalyticsFalseNegatives.feature_hash,
            )
        ).fetchall()
        false_negative_count_map = {}
        for feature_hash, false_negative_count in false_negative_counts_raw:
            false_negative_count_map[feature_hash] = false_negative_count
        positive_counts_raw = sess.exec(
            select(
                ProjectPredictionAnalytics.feature_hash,
                sql_sum(
                    (
                        (ProjectPredictionAnalytics.match_duplicate_iou < iou) & (ProjectPredictionAnalytics.iou >= iou)
                    ).cast(  # type: ignore
                        Integer
                    )  # type: ignore
                ),  # type: ignore
                sql_count(),
            )
            .where(ProjectPredictionAnalytics.prediction_hash == prediction_hash)
            .group_by(
                ProjectPredictionAnalytics.feature_hash,
            )
        )
        true_positive_count_map = {}
        false_positive_count_map = {}
        for feature_hash, tp_count, tp_fp_count in positive_counts_raw:
            fp_count = tp_fp_count - tp_count
            true_positive_count_map[feature_hash] = tp_count
            false_positive_count_map[feature_hash] = fp_count

        all_feature_hashes = (
            set(true_positive_count_map.keys())
            | set(false_negative_count_map.keys())
            | set(false_negative_all_feature_hashes)
        )

        # Calculated Average Recall
        # Mean Average Precision is calculated later
        feature_properties = {}
        for feature_hash in all_feature_hashes:
            fn_feature = false_negative_count_map.get(feature_hash, 0)
            tp_feature = true_positive_count_map.get(feature_hash, 0)
            fp_feature = false_positive_count_map.get(feature_hash, 0)
            feature_p = tp_feature / max(tp_feature + fp_feature, 1)
            feature_r = tp_feature / max(tp_feature + fn_feature, 1)
            feature_labels = tp_feature + fn_feature

            ar_values = []
            iou_thresholds = [x / 10.0 for x in range(5, 11, 1)]
            for iou_threshold in iou_thresholds:
                tp_iou: int = (
                    sess.exec(
                        select(sql_count()).where(
                            ProjectPredictionAnalytics.prediction_hash == prediction_hash,
                            ProjectPredictionAnalytics.feature_hash == feature_hash,
                            ProjectPredictionAnalytics.match_duplicate_iou < iou_threshold,
                            ProjectPredictionAnalytics.iou >= iou_threshold,
                        )
                    ).first()
                    or 0
                )
                iou_recall = tp_iou / max(feature_labels, 1)
                ar_values.append(iou_recall)
            feature_properties[feature_hash] = PredictionSummaryFeatureResult(
                ap=float("nan"),  # Assigned later
                ar=2.0 * sklearn_metrics.auc(iou_thresholds, ar_values),
                p=feature_p,
                r=feature_r,
                f1=(2 * feature_p * feature_r) / max(feature_p + feature_r, 1),
                tp=tp_feature,
                fp=fp_feature,
                fn=fn_feature,
            )

    # Calculate correlation for metrics
    metric_names = [key for key, value in AnnotationMetrics.items()]
    if engine.dialect == "sqlite":
        # Horrible implementation for use by sqlite.
        # FIXME: sqlite does not have corr(), stdev() or sqrt() functions, so have to implement manually.
        metric_stats = sess.execute(
            text(
                f"""
                SELECT
                    e_score,
                    avg(
                        ((
                            iou >= :iou and
                            match_duplicate_iou < :iou
                        ) - e_score) * ((
                            iou >= :iou and
                            match_duplicate_iou < :iou
                        ) - e_score)
                    ) as var_score,
                    {",".join([
                    f"e_{metric_name},"
                    f"avg(({metric_name} - e_{metric_name}) * ({metric_name} - e_{metric_name})) "
                    f"as var_{metric_name}"
                    for metric_name in metric_names
                ])}
                FROM project_prediction_analytics, (
                    SELECT
                        avg(
                            iou >= :iou and
                            match_duplicate_iou < :iou
                        ) as e_score,
                        {",".join([
                    f"avg({metric_name}) as e_{metric_name}"
                    for metric_name in metric_names
                ])}
                     FROM project_prediction_analytics
                     WHERE prediction_hash = :prediction_hash
                )
                WHERE prediction_hash = :prediction_hash
                """
            ).bindparams(bindparam("prediction_hash", GUID), bindparam("iou", Float)),
            params={
                "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                "iou": iou,
            },
        ).first()
        metric_stat_names = [f"{ty}_{name}" for name in ["score"] + metric_names for ty in ["e", "var"]]
        valid_metric_stats = {}
        valid_metric_stat_name_set = set(metric_names)
        for stat_name, stat_value in zip(metric_stat_names, metric_stats or []):
            if stat_value is None:
                stat = stat_name[len(stat_name.split("_")[0]) + 1 :]
                if stat in valid_metric_stat_name_set:
                    valid_metric_stat_name_set.remove(stat)
            else:
                if stat_name.startswith("var_"):
                    valid_metric_stats[stat_name.replace("var_", "std_")] = math.sqrt(stat_value)
                else:
                    valid_metric_stats[stat_name] = stat_value
        valid_metric_stat_names = list(valid_metric_stat_name_set)
        if len(valid_metric_stat_names) > 0:
            correlation_result = sess.execute(
                text(
                    f"""
                    SELECT
                        {",".join([
                        f'''
                            avg(
                                (
                                    (
                                        iou >= :iou and
                                        match_duplicate_iou < :iou
                                    ) - :e_score
                                ) * ({metric_name} - :e_{metric_name})
                            ) / (:std_score * :std_{metric_name}) as corr_{metric_name}
                            '''
                        for metric_name in valid_metric_stat_names
                    ])}
                    FROM project_prediction_analytics
                    WHERE prediction_hash = :prediction_hash
                    """
                ),
                params={
                    "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                    "iou": iou,
                    **valid_metric_stats,
                },
            ).first()
            correlations = {name: value for name, value in zip(valid_metric_stat_names, correlation_result)}  # type: ignore
        else:
            correlations = {}
    else:
        correlation_result = (
            sess.execute(
                text(
                    f"""
                SELECT
                    {",".join(
                        [   
                        f'''
                        coalesce(corr(
                            cast(cast((iou >= :iou and match_duplicate_iou < :iou) as integer) as real),
                            {metric_name}
                        ), 0.0)
                        '''
                        for metric_name in metric_names
                        ]
                    )}
                FROM project_prediction_analytics
                WHERE prediction_hash = :prediction_hash
                """
                ),
                params={
                    "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                    "iou": iou,
                },
            ).first()
            or []
        )
        correlations = {name: remove_nan_inf(value, inf=1.0) for name, value in zip(metric_names, correlation_result)}

    # Precision recall curves
    prs = {}
    for feature_hash in all_feature_hashes:
        sql = text(
            """
            SELECT
                max(to_group.p) as p,
                round(cast(to_group.r as numeric(3, 2)), 2) as r
            FROM (
                SELECT
                    cast(tp_count as real) / max(tp_and_fp_count) as p,
                    coalesce(cast(tp_count as real) / :r_divisor, 0.0) as r
                FROM (
                    SELECT
                        row_number() OVER (
                            ORDER BY PR.metric_confidence DESC ROWS UNBOUNDED PRECEDING
                        ) AS tp_and_fp_count,
                        sum(cast(
                            (
                                PR.iou >= :iou and
                                PR.match_duplicate_iou < :iou
                            ) as integer
                        )) OVER (
                            ORDER BY PR.metric_confidence DESC ROWS UNBOUNDED PRECEDING
                        ) AS tp_count
                    FROM project_prediction_analytics PR
                    WHERE PR.prediction_hash = :prediction_hash
                    AND PR.feature_hash = :feature_hash
                    ORDER BY PR.metric_confidence DESC
                ) AS seq_scan
                GROUP BY tp_count
            ) as to_group
            GROUP BY r
            ORDER BY r DESC
            """
        ).bindparams(bindparam("prediction_hash", GUID), bindparam("iou", Float), bindparam("r_divisor", Float))
        pr_result_raw = sess.execute(
            sql,
            params={
                "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                "iou": iou,
                "feature_hash": char8.process_bind_param(feature_hash, engine.dialect),
                "r_divisor": true_positive_count_map.get(feature_hash, 0)
                + false_negative_count_map.get(feature_hash, 0),
            },
        ).fetchall()
        pr_result = [PredictionPRPoint(p=p, r=r) for p, r in pr_result_raw]
        ap_vals = []
        for recall_threshold in range(0, 11, 1):
            ap_vals.append(max([v.p for v in pr_result if v.r >= (recall_threshold / 10.0)], default=0.0))
        prs[feature_hash] = pr_result
        feature_properties[feature_hash].ap = sum(ap_vals) / max(len(ap_vals), 1)

    importance = {}
    for metric_name in AnnotationMetrics.keys():
        importance_data_query = select(  # type: ignore
            ProjectPredictionAnalytics.iou
            * ((ProjectPredictionAnalytics.iou >= iou) & (ProjectPredictionAnalytics.match_duplicate_iou < iou))
            .cast(Integer)  # type: ignore
            .cast(Float),
            coalesce(getattr(ProjectPredictionAnalytics, metric_name).cast(Float), 0.0),
        ).where(
            ProjectPredictionAnalytics.prediction_hash == prediction_hash,
            is_not(getattr(ProjectPredictionAnalytics, metric_name), None),
        )
        importance_data = sess.exec(importance_data_query).fetchall()
        if len(importance_data) == 0:
            continue
        importance_regression = sklearn_mutual_info_regression(
            np.nan_to_num(np.array([float(d[1]) for d in importance_data]).reshape(-1, 1)),
            np.nan_to_num(np.array([float(d[0]) for d in importance_data])),
            random_state=42,
        )
        importance[metric_name] = float(importance_regression[0])

    t_tp: int = sum(p.tp for p in feature_properties.values())
    t_fp: int = sum(p.fp for p in feature_properties.values())
    t_fn: int = sum(p.fn for p in feature_properties.values())
    classification_t_tn: int = num_frames - t_tp - t_fn
    classification_accuracy: float = (t_tp + classification_t_tn) / max(num_frames, 1)
    return PredictionSummaryResult(
        classification_only=is_classification,
        classification_tTN=classification_t_tn if is_classification else None,
        classification_accuracy=classification_accuracy if is_classification else None,
        num_frames=num_frames,
        mAP=sum(p.ap for p in feature_properties.values()) / max(len(feature_properties), 1),
        mAR=sum(p.ar for p in feature_properties.values()) / max(len(feature_properties), 1),
        mP=sum(p.p for p in feature_properties.values()) / max(len(feature_properties), 1),
        mR=sum(p.r for p in feature_properties.values()) / max(len(feature_properties), 1),
        mF1=sum(p.f1 for p in feature_properties.values()) / max(len(feature_properties), 1),
        tTP=t_tp,
        tFP=t_fp,
        tFN=t_fn,
        # Feature properties
        feature_properties=feature_properties,
        prs=prs,
        # Metric properties
        correlation=correlations,
        importance=importance,
    )


class QueryMetricPerformanceEntry(BaseModel):
    m: float
    a: float
    n: int


class QueryMetricPerformance(BaseModel):
    precision: Dict[str, List[QueryMetricPerformanceEntry]]
    fns: Dict[str, List[QueryMetricPerformanceEntry]]


@router.get("/metric_performance")
def prediction_metric_performance(
    project_hash: uuid.UUID,
    prediction_hash: uuid.UUID,
    iou: float,
    metric_name: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(100),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine),
) -> QueryMetricPerformance:
    where_tp_fp = search_query.search_filters(
        tables=TABLES_PREDICTION_TP_FP,
        base=ProjectPredictionAnalytics,
        search=filters,
        project_filters={
            "prediction_hash": [prediction_hash],
            # FIXME: needs project hash for correct data join behaviour!!
        },
    )
    where_fn = search_query.search_filters(
        tables=TABLES_ANNOTATION,
        base=ProjectPredictionAnalyticsFalseNegatives,
        search=filters,
        project_filters={
            "prediction_hash": [prediction_hash],
        },
    )
    with Session(engine) as sess:
        metric_attr = metric_query.get_metric_or_enum(
            engine=engine,
            table=ProjectPredictionAnalytics,
            attr_name=metric_name,
            metrics=AnnotationMetrics,
            enums={},
            buckets=buckets,  # FIXME: bucket behaviour needs improvement (sql round is better than current impl).
        )
        metric_buckets = sess.exec(
            select(
                ProjectPredictionAnalytics.feature_hash,
                metric_attr.group_attr,
                coalesce(
                    sql_sum(
                        (
                            (ProjectPredictionAnalytics.match_duplicate_iou < iou)
                            & (ProjectPredictionAnalytics.iou >= iou)
                        ).cast(  # type: ignore
                            Integer
                        )  # type: ignore
                    ),
                    0,
                ),  # type: ignore
                sql_count(),
            )
            .where(is_not(metric_attr.filter_attr, None), *where_tp_fp)
            .group_by(
                ProjectPredictionAnalytics.feature_hash,
                metric_attr.group_attr,
            )
            .order_by(
                metric_attr.group_attr,
            )
        ).fetchall()
        precision: Dict[str, List[QueryMetricPerformanceEntry]] = {}
        tp_count: Dict[Tuple[str, float], int] = {}
        for feature, bucket_m, bucket_tp, bucket_tp_fp in metric_buckets:
            bucket_avg = float(bucket_tp) / max(float(bucket_tp_fp), 1.0)
            precision.setdefault(feature, []).append(
                QueryMetricPerformanceEntry(m=bucket_m, a=bucket_avg, n=bucket_tp_fp)
            )
            tp_count[(feature, bucket_m)] = bucket_tp_fp

        metric_attr = metric_query.get_metric_or_enum(
            engine=engine,
            table=ProjectAnnotationAnalytics,
            attr_name=metric_name,
            metrics=AnnotationMetrics,
            enums={},
            buckets=buckets,
        )
        metric_fn_buckets = sess.exec(
            select(
                ProjectPredictionAnalyticsFalseNegatives.feature_hash,
                metric_attr.group_attr,
                sql_count(),
            )
            .where(
                ProjectPredictionAnalyticsFalseNegatives.iou_threshold < iou,
                ProjectAnnotationAnalytics.project_hash == project_hash,
                ProjectAnnotationAnalytics.du_hash == ProjectPredictionAnalyticsFalseNegatives.du_hash,
                ProjectAnnotationAnalytics.frame == ProjectPredictionAnalyticsFalseNegatives.frame,
                is_not(metric_attr.filter_attr, None),
                *where_fn,
            )
            .group_by(
                ProjectPredictionAnalyticsFalseNegatives.feature_hash,
                metric_attr.group_attr,
            )
            .order_by(
                metric_attr.group_attr,
            )
        ).fetchall()
        fns: Dict[str, List[QueryMetricPerformanceEntry]] = {}
        for feature, bucket_m, fn_num in metric_fn_buckets:
            fp_tp_num = tp_count.get((feature, bucket_m), 0)
            label_num = fn_num + fp_tp_num
            fns.setdefault(feature, []).append(
                QueryMetricPerformanceEntry(m=bucket_m, a=0 if label_num == 0 else fn_num / label_num, n=label_num)
            )
    return QueryMetricPerformance(
        precision=precision,
        fns=fns,
    )


class PredictionItem(BaseModel):
    annotation_metrics: Dict[str, Dict[str, Optional[float]]]
    objects: list[dict]
    classifications: list[dict]
    label_hash: uuid.UUID


def _sanitise_nan(value: Optional[float]) -> Optional[float]:
    if value is None or math.isnan(value):
        return None
    return value


@router.get("/preview/{data_item}")
def get_prediction_item(
    project_hash: uuid.UUID,
    prediction_hash: uuid.UUID,
    item_details: DataItem = Depends(parse_data_item),
    engine: Engine = Depends(dep_engine),
) -> PredictionItem:
    du_hash = item_details.du_hash
    frame = item_details.frame
    with Session(engine) as sess:
        du_meta = sess.exec(
            select(ProjectPredictionDataUnitMetadata).where(
                ProjectPredictionDataUnitMetadata.project_hash == project_hash,
                ProjectPredictionDataUnitMetadata.prediction_hash == prediction_hash,
                ProjectPredictionDataUnitMetadata.du_hash == du_hash,
                ProjectPredictionDataUnitMetadata.frame == frame,
            )
        ).first()
        if du_meta is None:
            raise ValueError(f"Invalid request: du_hash={du_hash}, frame={frame} is missing DataUnitMeta")
        data_meta = sess.exec(
            select(ProjectPredictionDataMetadata).where(
                ProjectPredictionDataMetadata.project_hash == project_hash,
                ProjectPredictionDataMetadata.prediction_hash == prediction_hash,
                ProjectPredictionDataMetadata.data_hash == du_meta.data_hash,
            )
        ).first()
        if data_meta is None:
            raise ValueError(f"Invalid request: du_hash={du_hash}, frame={frame} is missing DataMeta")
        annotation_analytics_list = sess.exec(
            select(ProjectPredictionAnalytics).where(
                ProjectPredictionAnalytics.project_hash == project_hash,
                ProjectPredictionAnalytics.prediction_hash == prediction_hash,
                ProjectPredictionAnalytics.du_hash == du_hash,
                ProjectPredictionAnalytics.frame == frame,
            )
        ).fetchall()

    return PredictionItem(
        annotation_metrics={
            annotation_analytics.annotation_hash: {
                k: _sanitise_nan(getattr(annotation_analytics, k)) for k in AnnotationMetrics
            }
            for annotation_analytics in annotation_analytics_list
        },
        objects=du_meta.objects,
        classifications=du_meta.classifications,
        label_hash=data_meta.label_hash,
    )
