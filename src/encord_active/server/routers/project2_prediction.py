import functools
import math
import uuid
import json
from enum import Enum
from typing import Optional, Dict, Tuple, Union, List, Literal, Type

import numpy as np
from fastapi import APIRouter
from pynndescent import NNDescent
from sklearn.feature_selection import mutual_info_regression
from sqlalchemy import func, bindparam, Float, text
from sqlalchemy.sql.operators import is_not, in_op, not_between_op, between_op
from sqlmodel import Session, select
from sqlmodel.sql.sqltypes import GUID

from encord_active.db.metrics import DataMetrics, AnnotationMetrics, MetricDefinition, MetricType
from encord_active.db.models import ProjectDataAnalytics, ProjectAnnotationAnalytics, AnnotationType, \
    ProjectPredictionObjectResults
from encord_active.server.routers.project2_engine import engine

router = APIRouter(
    prefix="/get/{project_hash}/predictions/get/{prediction_hash}",
)


# FIXME: verify project_hash <=> prediction hash is allowed to access

@router.get("/summary")
def get_project_prediction_summary(prediction_hash: uuid.UUID, iou: float):
    guid = GUID()
    with Session(engine) as sess:
        # FIXME: check that feature_hash compare can be removed with changes to pre-calculation of match_iou condition.
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
                                    AP.match_duplicate_iou < :iou
                                ) as integer
                            )) OVER (
                                ORDER BY AP.confidence DESC ROWS UNBOUNDED PRECEDING
                            ) AS ap_positives
                        FROM active_project_prediction_results AP
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
                                    AR.match_duplicate_iou < :iou
                                ) as integer
                            )) OVER (
                                ORDER BY AR.iou DESC ROWS UNBOUNDED PRECEDING
                            ) AS ar_positives
                        FROM active_project_prediction_results AR
                        WHERE AR.prediction_hash = FG.prediction_hash
                        AND AR.feature_hash = FG.feature_hash
                        ORDER BY AR.iou
                    )
                ) as ar
                FROM active_project_prediction_results FG
                WHERE FG.prediction_hash = :prediction_hash
                GROUP BY FG.feature_hash
            )
            """
        ).bindparams(bindparam("prediction_hash", GUID), bindparam("iou", Float))
        precision_recall = sess.execute(
            sql,
            params={
                "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                "iou": iou
            },
        ).fetchall()

        # Calculate correlation for metrics
        # FIXME: sqlite does not have corr(), stdev() or sqrt() functions, so have to implement manually.
        metric_names = [
            key
            for key, value in AnnotationMetrics.items()
            if value.virtual is None
        ]
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
                FROM active_project_prediction_results, (
                    SELECT
                        avg(
                            iou >= :iou and
                            match_duplicate_iou < :iou
                        ) as e_score,
                        {",".join([
                    f"avg({metric_name}) as e_{metric_name}"
                    for metric_name in metric_names
                ])}
                     FROM active_project_prediction_results
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
        metric_stat_names = [
            f"{ty}_{name}"
            for name in ["score"] + metric_names
            for ty in ["e", "var"]
        ]
        valid_metric_stats = {}
        valid_metric_stat_name_set = set(metric_names)
        for stat_name, stat_value in zip(metric_stat_names, metric_stats):
            if stat_value is None:
                stat = stat_name[len(stat_name.split("_")[0]) + 1:]
                if stat in valid_metric_stat_name_set:
                    valid_metric_stat_name_set.remove(stat)
            else:
                if stat_name.startswith("var_"):
                    valid_metric_stats[stat_name.replace("var_", "std_")] = math.sqrt(stat_value)
                else:
                    valid_metric_stats[stat_name] = stat_value
        valid_metric_stat_names = list(valid_metric_stat_name_set)
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
                FROM active_project_prediction_results
                WHERE prediction_hash = :prediction_hash
                """
            ),
            params={
                "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                "iou": iou,
                **valid_metric_stats
            },
        ).first()
        correlations = {
            name: value
            for name, value in zip(valid_metric_stat_names, correlation_result)
        }

        # Precision recall curves
        prs = {}
        for pr_summary in precision_recall:
            feature_hash = pr_summary["feature_hash"]
            sql = text(
                """
                SELECT
                    max(p) as p,
                    floor(r * 50) / 50 as r
                FROM (
                    SELECT
                        cast(tp_count as real) / tp_and_fp_count as p,
                        cast(tp_count as real) / (
                            (
                                SELECT COUNT(*) FROM active_project_prediction_results PM
                                WHERE PM.prediction_hash = :prediction_hash
                                AND PM.feature_hash = :feature_hash
                                AND PM.iou >= :iou
                                AND PM.match_duplicate_iou < :iou
                            ) + (
                                SELECT COUNT(*) FROM active_project_prediction_unmatched UM
                                WHERE UM.prediction_hash = :prediction_hash
                                AND UM.feature_hash == :feature_hash
                            )
                        ) as r
                    FROM (
                        SELECT
                            row_number() OVER (
                                ORDER BY PR.confidence DESC ROWS UNBOUNDED PRECEDING
                            ) AS tp_and_fp_count,
                            sum(cast(
                                (
                                    PR.iou >= :iou and
                                    PR.match_duplicate_iou < :iou
                                ) as integer
                            )) OVER (
                                ORDER BY PR.confidence DESC ROWS UNBOUNDED PRECEDING
                            ) AS tp_count
                        FROM active_project_prediction_results PR
                        WHERE PR.prediction_hash = :prediction_hash
                        AND PR.feature_hash = :feature_hash
                        ORDER BY confidence DESC
                    )
                    GROUP BY tp_count
                )
                GROUP BY floor(r * 50) / 50
                ORDER BY floor(r * 50) / 50 DESC
                """
            ).bindparams(bindparam("prediction_hash", GUID), bindparam("iou", Float))
            pr_result = sess.execute(
                sql,
                params={
                    "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                    "iou": iou,
                    "feature_hash": feature_hash
                },
            ).fetchall()
            prs[feature_hash] = pr_result

            importance = {}
            for metric_name in AnnotationMetrics.keys():
                importance_data_query = select(
                    ProjectPredictionObjectResults.iou * (
                            (ProjectPredictionObjectResults.iou >= iou) &
                            (ProjectPredictionObjectResults.match_duplicate_iou < iou)
                    ),
                    getattr(ProjectPredictionObjectResults, metric_name)
                ).where(
                    ProjectPredictionObjectResults.prediction_hash == prediction_hash,
                    is_not(getattr(ProjectPredictionObjectResults, metric_name), None)
                )
                importance_data = sess.exec(importance_data_query).fetchall()
                if len(importance_data) == 0:
                    continue
                importance_regression = mutual_info_regression(
                    np.array([float(d[1]) for d in importance_data]).reshape(-1, 1),
                    np.array([float(d[0]) for d in importance_data]),
                    random_state=42
                )
                importance[metric_name] = float(importance_regression[0])

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
        "correlation": correlations,
        "importance": importance,
        "prs": prs
    }


@router.get("/metric_performance")
def prediction_metric_performance(
        prediction_hash: uuid.UUID, buckets: int, iou: float, metric_name: str
):
    with Session(engine) as sess:
        metric = AnnotationMetrics[metric_name]
        metric_attr = getattr(ProjectPredictionObjectResults, metric_name)
        where = [
            ProjectPredictionObjectResults.prediction_hash == prediction_hash,
            is_not(metric_attr, None)
        ]
        if metric.type == MetricType.NORMAL:
            metric_min = 0.0
            metric_max = 1.0
        else:
            metric_min = sess.exec(select(func.min(metric_attr)).where(*where)).first()
            metric_max = sess.exec(select(func.max(metric_attr)).where(*where)).first()
            if metric_min is None or metric_max is None:
                return {
                    "precision": {},
                    "fns": {},
                }
            metric_min = float(metric_min)
            metric_max = float(metric_max)
        metric_bucket_size = (metric_max - metric_min) / buckets
        metric_buckets = sess.exec(
            select(
                ProjectPredictionObjectResults.feature_hash,
                func.min(func.floor((metric_attr - metric_min) / metric_bucket_size), buckets - 1),
                func.avg(
                    (ProjectPredictionObjectResults.iou >= iou) &
                    (ProjectPredictionObjectResults.match_duplicate_iou < iou)
                ),
                func.count()
            ).where(
                *where,
            ).group_by(
                func.min(func.floor((metric_attr - metric_min) / metric_bucket_size), buckets - 1),
                ProjectPredictionObjectResults.feature_hash
            )
        )
        precision = {}
        for feature, bucket_id, bucket_avg, bucket_num in metric_buckets:
            precision.setdefault(feature, []).append({
                "m": metric_min + (metric_bucket_size * int(bucket_num)),
                "a": bucket_avg,
                "n": bucket_num
            })

    return {
        "precision": precision,
        "fns": {},
    }
