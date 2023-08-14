import json
import math
import uuid
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sklearn.feature_selection import mutual_info_regression
from sqlalchemy import Float, Integer, bindparam, text
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
)
from encord_active.server.routers.project2_engine import engine
from encord_active.server.routers.queries import metric_query, search_query
from encord_active.server.routers.queries.domain_query import (
    TABLES_PREDICTION_FN,
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


def check_project_prediction_hash_match(project_hash: uuid.UUID, prediction_hash: uuid.UUID):
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
    prefix="/{project_hash}/predictions/{prediction_hash}", dependencies=[Depends(check_project_prediction_hash_match)]
)


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


@router.get("/summary")
def get_project_prediction_summary(
    prediction_hash: uuid.UUID,
    iou: float,
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
):
    # FIXME: this command will return the wrong answers when filters are applied!!!
    tp_fp_where = search_query.search_filters(
        tables=TABLES_PREDICTION_TP_FP,
        base="analytics",
        search=filters,
        project_filters={
            "prediction_hash": [prediction_hash],
            # FIXME: project_hash
        },
    )
    # FIXME: fn_where needs separate set of queries as this is implemented as a side table join.
    # FIXME: performance improvements are possible
    guid = GUID()
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
        num_frames = sess.exec(
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
        total_false_negative_count = 0
        for feature_hash, false_negative_count in false_negative_counts_raw:
            total_false_negative_count += false_negative_count
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
        total_true_positive_count = 0
        total_false_positive_count = 0
        for feature_hash, tp_count, tp_fp_count in positive_counts_raw:
            fp_count = tp_fp_count - tp_count
            true_positive_count_map[feature_hash] = tp_count
            false_positive_count_map[feature_hash] = fp_count
            total_true_positive_count += tp_count
            total_false_positive_count += fp_count

        all_feature_hashes = (
            set(true_positive_count_map.keys())
            | set(false_negative_count_map.keys())
            | set(false_negative_all_feature_hashes)
        )

        # FIXME: check that feature_hash compare can be removed with changes to pre-calculation of match_iou condition.
        precision_recall = {}
        for feature_hash in all_feature_hashes:
            sql = text(
                """
                SELECT (
                    SELECT sum(
                        coalesce(cast(ap_inner.ap_positives as real) / cast(ap_inner.ap_total as real), 0.0)
                    ) / (
                        count(*)
                    ) as ap FROM (
                        SELECT
                            row_number() OVER (
                                ORDER BY AP.metric_confidence DESC ROWS UNBOUNDED PRECEDING
                            ) AS ap_total,
                            sum(cast(
                                (
                                    AP.iou >= :iou and
                                    AP.match_duplicate_iou < :iou
                                ) as integer
                            )) OVER (
                                ORDER BY AP.metric_confidence DESC ROWS UNBOUNDED PRECEDING
                            ) AS ap_positives
                        FROM active_project_prediction_analytics AP
                        WHERE AP.prediction_hash = :prediction_hash
                        AND AP.feature_hash = :feature_hash
                        ORDER BY AP.metric_confidence DESC
                    ) as ap_inner
                ),
                (
                    SELECT sum(
                        coalesce(cast(ar_inner.ar_positives as real) / cast(ar_inner.ar_total as real), 0.0)
                    ) / (
                        count(*) + :false_negatives
                    ) FROM (
                        SELECT
                            row_number() OVER (
                                ORDER BY AR.iou DESC ROWS UNBOUNDED PRECEDING
                            ) AS ar_total,
                            sum(cast(
                                (
                                    AR.iou >= :iou and
                                    AR.match_duplicate_iou < :iou
                                ) as integer
                            )) OVER (
                                ORDER BY AR.iou DESC ROWS UNBOUNDED PRECEDING
                            ) AS ar_positives
                        FROM active_project_prediction_analytics AR
                        WHERE AR.prediction_hash = :prediction_hash
                        AND AR.feature_hash = :feature_hash
                        ORDER BY AR.iou
                    ) as ar_inner
                )
                """
            ).bindparams(bindparam("prediction_hash", GUID), bindparam("iou", Float))
            pr_result = sess.execute(
                sql,
                params={
                    "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                    "iou": iou,
                    "feature_hash": feature_hash,
                    "false_negatives": false_negative_count_map.get(feature_hash, 0),
                },
            ).first()
            precision_recall[feature_hash] = {
                # FIXME: NULL precision / recall handling on the backend!?!
                "ap": pr_result[0] or 0.0,  # type: ignore
                "ar": pr_result[1] or 0.0,  # type: ignore
            }

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
                FROM active_project_prediction_analytics, (
                    SELECT
                        avg(
                            iou >= :iou and
                            match_duplicate_iou < :iou
                        ) as e_score,
                        {",".join([
                    f"avg({metric_name}) as e_{metric_name}"
                    for metric_name in metric_names
                ])}
                     FROM active_project_prediction_analytics
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
                    FROM active_project_prediction_analytics
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
                FROM active_project_prediction_analytics
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
                    cast(tp_count as real) / :r_divisor as r
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
                    FROM active_project_prediction_analytics PR
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
        pr_result = sess.execute(
            sql,
            params={
                "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                "iou": iou,
                "feature_hash": feature_hash,
                "r_divisor": true_positive_count_map.get(feature_hash, 0)
                + false_negative_count_map.get(feature_hash, 0),
            },
        ).fetchall()
        prs[feature_hash] = pr_result

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
        importance_regression = mutual_info_regression(
            np.nan_to_num(np.array([float(d[1]) for d in importance_data]).reshape(-1, 1)),
            np.nan_to_num(np.array([float(d[0]) for d in importance_data])),
            random_state=42,
        )
        importance[metric_name] = float(importance_regression[0])

    v = {
        "classification_only": is_classification,
        "num_frames": num_frames or 0,
        "mAP": sum(pr["ap"] for pr in precision_recall.values()) / max(len(precision_recall), 1),
        "mAR": sum(pr["ar"] for pr in precision_recall.values()) / max(len(precision_recall), 1),
        "precisions": {feature_hash: pr["ap"] for feature_hash, pr in precision_recall.items()},
        "recalls": {feature_hash: pr["ar"] for feature_hash, pr in precision_recall.items()},
        "correlation": correlations,
        "importance": importance,
        "prs": prs,
        "tTP": total_true_positive_count,
        "tFP": total_false_positive_count,
        "tFN": total_false_negative_count,
        "tp": true_positive_count_map,
        "fp": false_positive_count_map,
        "fn": false_negative_count_map,
    }
    print(v)
    return v


@router.get("/analytics/{prediction_domain}/distribution")
def prediction_metric_distribution(
    prediction_hash: uuid.UUID,
    group: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(100),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
) -> metric_query.QueryDistribution:
    # FIXME: prediction_domain!!! (this & scatter) both need it implemented
    # FIXME: how to do we want to support fp-tp-fn split (2-group, 3-group)?
    # FIXME:  or try to support unified?
    with Session(engine) as sess:
        return metric_query.query_attr_distribution(
            sess=sess,
            tables=TABLES_PREDICTION_TP_FP,
            project_filters={
                "prediction_hash": [prediction_hash],
                # FIXME: needs project_hash
            },
            attr_name=group,
            buckets=buckets,
            filters=filters,
        )


@router.get("/analytics/{prediction_domain}/scatter")
def prediction_metric_scatter(
    prediction_hash: uuid.UUID,
    x_metric: str,
    y_metric: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(10),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
) -> metric_query.QueryScatter:
    with Session(engine) as sess:
        return metric_query.query_attr_scatter(
            sess=sess,
            tables=TABLES_PREDICTION_TP_FP,
            project_filters={
                "prediction_hash": [prediction_hash],
                # FIXME: needs project_hash
            },
            x_metric_name=x_metric,
            y_metric_name=y_metric,
            buckets=buckets,
            filters=filters,
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
) -> QueryMetricPerformance:
    where_tp_fp = search_query.search_filters(
        tables=TABLES_PREDICTION_TP_FP,
        base="analytics",
        search=filters,
        project_filters={
            "prediction_hash": [prediction_hash],
            # FIXME: needs project hash for correct data join behaviour!!
        },
    )
    where_fn = search_query.search_filters(
        tables=TABLES_PREDICTION_FN,
        base="analytics",
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


@router.get("/search")
def prediction_search(
    iou: float,
    metric_filters: str,
    enum_filters: str,
):
    metric_filters_dict = json.loads(metric_filters)
    enum_filters_dict = json.loads(enum_filters)
    # Some enums have special meaning.
    include_prediction_tp = True
    include_prediction_fp = True
    include_prediction_fn = True
    if "prediction_type" in enum_filters_dict:
        filters = set(enum_filters_dict.pop("prediction_type"))
        include_prediction_tp = "tp" in filters
        include_prediction_fp = "fp" in filters
        include_prediction_fn = "fn" in filters
    include_prediction_hashes = enum_filters_dict.pop("prediction_hash", None)
    include_data = True
    include_annotations = True

    return None


@router.get("/preview/{du_hash/{frame}/{annotation_hash}")
def get_prediction_item(
    du_hash: uuid.UUID,
    frame: int,
    annotation_hash: Optional[str] = None,
):
    # FIXME: return required metadata to correctly view the associated value.
    return None
