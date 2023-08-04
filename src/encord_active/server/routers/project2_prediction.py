import json
import math
import uuid
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import APIRouter
from sklearn import metrics as sklearn_metrics
from sklearn.feature_selection import (
    mutual_info_regression as sklearn_mutual_info_regression,
)
from sqlalchemy import Float, Integer, bindparam, text
from sqlalchemy.sql.operators import is_not
from sqlmodel import Session, select
from sqlmodel.sql.sqltypes import GUID

from encord_active.db.enums import AnnotationType
from encord_active.db.metrics import AnnotationMetrics
from encord_active.db.models import (
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

router = APIRouter(
    prefix="/{project_hash}/predictions/{prediction_hash}",
)


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
            feature_properties[feature_hash] = {
                "ar": 2.0 * sklearn_metrics.auc(iou_thresholds, ar_values),
                "p": feature_p,
                "r": feature_r,
                "f1": (2 * feature_p * feature_r) / max(feature_p + feature_r, 1),
                "tp": tp_feature,
                "fp": fp_feature,
                "fn": fn_feature,
            }

    # Calculate correlation for metrics
    # FIXME: sqlite does not have corr(), stdev() or sqrt() functions, so have to implement manually.
    metric_names = [key for key, value in AnnotationMetrics.items()]
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

    # Precision recall curves
    prs = {}
    for feature_hash in all_feature_hashes:
        sql = text(
            """
            SELECT
                max(p) as p,
                round(r, 2) as r
            FROM (
                SELECT
                    cast(tp_count as real) / max(tp_and_fp_count) as p,
                    coalesce(cast(tp_count as real) / :r_divisor, 0.0) as r
                FROM (
                    SELECT
                        row_number() OVER (
                            ORDER BY PR.metric_label_confidence DESC ROWS UNBOUNDED PRECEDING
                        ) AS tp_and_fp_count,
                        sum(cast(
                            (
                                PR.iou >= :iou and
                                PR.match_duplicate_iou < :iou
                            ) as integer
                        )) OVER (
                            ORDER BY PR.metric_label_confidence DESC ROWS UNBOUNDED PRECEDING
                        ) AS tp_count
                    FROM active_project_prediction_analytics PR
                    WHERE PR.prediction_hash = :prediction_hash
                    AND PR.feature_hash = :feature_hash
                    ORDER BY metric_label_confidence DESC
                )
                GROUP BY tp_count
            )
            GROUP BY round(r, 2)
            ORDER BY round(r, 2) DESC
            """
        ).bindparams(bindparam("prediction_hash", GUID), bindparam("iou", Float), bindparam("r_divisor", Float))
        pr_result_raw = sess.execute(
            sql,
            params={
                "prediction_hash": guid.process_bind_param(prediction_hash, engine.dialect),
                "iou": iou,
                "feature_hash": feature_hash,
                "r_divisor": true_positive_count_map.get(feature_hash, 0)
                + false_negative_count_map.get(feature_hash, 0),
            },
        ).fetchall()
        pr_result = [{"p": p, "r": r} for p, r in pr_result_raw]
        ap_vals = []
        for recall_threshold in range(0, 11, 1):
            ap_vals.append(max([v["p"] for v in pr_result if v["r"] >= (recall_threshold / 10.0)], default=0.0))
        prs[feature_hash] = pr_result
        feature_properties[feature_hash]["ap"] = sum(ap_vals) / max(len(ap_vals), 1)

    importance = {}
    for metric_name in AnnotationMetrics.keys():
        importance_data_query = select(
            ProjectPredictionAnalytics.iou
            * ((ProjectPredictionAnalytics.iou >= iou) & (ProjectPredictionAnalytics.match_duplicate_iou < iou)),
            getattr(ProjectPredictionAnalytics, metric_name),
        ).where(
            ProjectPredictionAnalytics.prediction_hash == prediction_hash,
            is_not(getattr(ProjectPredictionAnalytics, metric_name), None),
        )
        importance_data = sess.exec(importance_data_query).fetchall()
        if len(importance_data) == 0:
            continue
        importance_regression = sklearn_mutual_info_regression(
            np.array([float(d[1]) for d in importance_data]).reshape(-1, 1),
            np.array([float(d[0]) for d in importance_data]),
            random_state=42,
        )
        importance[metric_name] = float(importance_regression[0])

    t_tp: int = sum(p["tp"] for p in feature_properties.values())
    t_fp: int = sum(p["fp"] for p in feature_properties.values())
    t_fn: int = sum(p["fn"] for p in feature_properties.values())
    classification_t_tn: int = num_frames - t_tp - t_fn
    classification_accuracy: float = (t_tp + classification_t_tn) / max(num_frames, 1)
    return {
        "classification_only": is_classification,
        "classification_tTN": classification_t_tn if is_classification else None,
        "classification_accuracy": classification_accuracy if is_classification else None,
        "num_frames": num_frames,
        "mAP": sum(p["ap"] for p in feature_properties.values()) / max(len(feature_properties), 1),
        "mAR": sum(p["ar"] for p in feature_properties.values()) / max(len(feature_properties), 1),
        "mP": sum(p["p"] for p in feature_properties.values()) / max(len(feature_properties), 1),
        "mR": sum(p["r"] for p in feature_properties.values()) / max(len(feature_properties), 1),
        "mF1": sum(p["f1"] for p in feature_properties.values()) / max(len(feature_properties), 1),
        "tTP": t_tp,
        "tFP": t_fp,
        "tFN": t_fn,
        # Feature properties
        "feature_properties": feature_properties,
        "prs": prs,
        # Metric properties
        "correlation": correlations,
        "importance": importance,
    }


@router.get("/analytics/{prediction_domain}/distribution")
def prediction_metric_distribution(
    prediction_hash: uuid.UUID,
    group: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(100),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
):
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
):
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


@router.get("/metric_performance")
def prediction_metric_performance(
    prediction_hash: uuid.UUID,
    iou: float,
    metric_name: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(100),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
):
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
                sql_sum(
                    (
                        (ProjectPredictionAnalytics.match_duplicate_iou < iou) & (ProjectPredictionAnalytics.iou >= iou)
                    ).cast(  # type: ignore
                        Integer
                    )  # type: ignore
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
        precision: Dict[str, List[dict]] = {}
        tp_count: Dict[Tuple[str, float], int] = {}
        for feature, bucket_m, bucket_tp, bucket_tp_fp in metric_buckets:
            bucket_avg = float(bucket_tp) / float(bucket_tp_fp)
            precision.setdefault(feature, []).append({"m": bucket_m, "a": bucket_avg, "n": bucket_tp_fp})
            tp_count[(feature, bucket_m)] = bucket_tp_fp

        metric_attr = metric_query.get_metric_or_enum(
            table=ProjectPredictionAnalyticsFalseNegatives,  # type: ignore
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
        fns: Dict[str, List[dict]] = {}
        for feature, bucket_m, fn_num in metric_fn_buckets:
            fp_tp_num = tp_count.get((feature, bucket_m), 0)
            label_num = fn_num + fp_tp_num
            fns.setdefault(feature, []).append(
                {"m": bucket_m, "a": 0 if label_num == 0 else fn_num / label_num, "n": label_num}
            )
    return {
        "precision": precision,
        "fns": fns,
    }


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
