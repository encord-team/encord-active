import math
import uuid
from enum import Enum
from typing import Literal, List, Optional, Tuple, Dict, Type, Union

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import Integer, desc as desc_fn, literal
from sqlmodel import Session, select
from sqlalchemy.engine import Engine

from encord_active.db.models import (
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsFalseNegatives,
    ProjectAnnotationAnalytics,
    ProjectPredictionAnalyticsReduced,
)
from encord_active.server.dependencies import dep_engine
from encord_active.server.routers.project2_analysis import AnalysisSearch
from encord_active.server.routers.queries import search_query, metric_query
from encord_active.server.routers.queries.domain_query import TABLES_ANNOTATION, TABLES_PREDICTION_TP_FP
from encord_active.server.routers.queries.metric_query import literal_bucket_depends
from encord_active.server.routers.queries.search_query import SearchFiltersFastAPIDepends


class PredictionDomain(Enum):
    ALL = "a"
    POSITIVE = "p"
    TRUE_POSITIVE = "tp"
    FALSE_POSITIVE = "fp"
    FALSE_NEGATIVE = "fn"


def _fn_extra_where(project_hash: uuid.UUID, prediction_hash: uuid.UUID) -> list:
    return [
        ProjectPredictionAnalyticsFalseNegatives.project_hash == project_hash,
        ProjectPredictionAnalyticsFalseNegatives.prediction_hash == prediction_hash,
        ProjectPredictionAnalyticsFalseNegatives.du_hash == ProjectAnnotationAnalytics.du_hash,
        ProjectPredictionAnalyticsFalseNegatives.frame == ProjectAnnotationAnalytics.frame,
        ProjectPredictionAnalyticsFalseNegatives.annotation_hash == ProjectAnnotationAnalytics.annotation_hash,
    ]


def _tp_fp_extra_where(
    domain: PredictionDomain,
    iou: float,
    table: Type[Union[ProjectPredictionAnalytics, ProjectPredictionAnalyticsReduced]],
    prediction_hash: uuid.UUID,
) -> list:
    tp_cond = (ProjectPredictionAnalytics.iou >= iou) & (ProjectPredictionAnalytics.match_duplicate_iou < iou)
    tp_where = None
    if domain == PredictionDomain.TRUE_POSITIVE:
        tp_where = tp_cond
    elif domain == PredictionDomain.FALSE_POSITIVE:
        tp_where = ~tp_cond
    if tp_where is None:
        return []
    elif table == ProjectPredictionAnalytics:
        return [tp_where]
    else:
        return [
            tp_where,
            ProjectPredictionAnalytics.prediction_hash == prediction_hash,
            ProjectPredictionAnalytics.du_hash == table.du_hash,
            ProjectPredictionAnalytics.frame == table.frame,
            ProjectPredictionAnalytics.annotation_hash == table.annotation_hash,
        ]


def _tp_int(iou: float) -> int:
    return ((ProjectPredictionAnalytics.iou >= iou) & (ProjectPredictionAnalytics.match_duplicate_iou < iou)).cast(
        Integer
    )


router = APIRouter(
    prefix="/analytics/{domain}",
)


class PredictionQueryScatterPoint(BaseModel):
    x: float
    y: float
    n: int
    tp: int
    fp: int
    fn: int


class PredictionQuery2DEmbedding(BaseModel):
    count: int
    reductions: List[PredictionQueryScatterPoint]


@router.get("/reductions/{reduction_hash}/summary")
def route_prediction_reduction_scatter(
    project_hash: uuid.UUID,
    prediction_hash: uuid.UUID,
    domain: PredictionDomain,
    iou: float,
    reduction_hash: uuid.UUID,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(10),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine),
) -> PredictionQuery2DEmbedding:
    # Where conditions for FN table
    with Session(engine) as sess:
        tp_fp_select = None
        if domain != PredictionDomain.FALSE_NEGATIVE:
            tp_fp_select = metric_query.select_for_query_reduction_scatter(
                dialect=sess.bind.dialect,
                tables=TABLES_PREDICTION_TP_FP,
                project_filters={
                    "prediction_hash": [prediction_hash],
                    "reduction_hash": [reduction_hash],
                },
                buckets=buckets,
                filters=filters,
                extra_where=_tp_fp_extra_where(
                    domain,
                    iou,
                    ProjectPredictionAnalyticsReduced,
                    prediction_hash,
                ),
                extra_select=(
                    metric_query.sql_sum(_tp_int(iou)).label("tp"),
                    literal(0).label("fn"),
                ),
            )
        fn_select = None
        if domain in {PredictionDomain.FALSE_NEGATIVE, PredictionDomain.ALL}:
            fn_select = metric_query.select_for_query_reduction_scatter(
                dialect=sess.bind.dialect,
                tables=TABLES_ANNOTATION,
                project_filters={
                    "prediction_hash": [prediction_hash],
                    "project_hash": [project_hash],
                    "reduction_hash": [reduction_hash],
                },
                buckets=buckets,
                filters=filters,
                extra_where=_fn_extra_where(project_hash, prediction_hash),
                extra_select=(literal(0).label("tp"), metric_query.sql_count().label("fn")),
            )
    if fn_select is not None and tp_fp_select is not None:
        query_select = (
            select(
                "xv",
                "yv",
                metric_query.sql_sum("n"),
                metric_query.sql_sum("tp"),
                metric_query.sql_sum("fn"),
            )
            .from_statement(fn_select.union_all(tp_fp_select))
            .group_by("xv", "yv")
        )
    elif fn_select is not None:
        query_select = fn_select
    elif tp_fp_select is not None:
        query_select = tp_fp_select
    else:
        raise RuntimeError("Bug in prediction reduction")

    with Session(engine) as sess:
        print(f"Project reduction embedding: {query_select}")
        results = sess.exec(query_select).fetchall()
        print(f"Project reduction results: {results}")

    return PredictionQuery2DEmbedding(
        count=sum(n for x, y, n, tp, fn in results),
        reductions=[
            PredictionQueryScatterPoint(
                x=x if not math.isnan(x) else 0, y=y if not math.isnan(y) else 0, n=n, tp=tp, fp=(n - tp - fn), fn=fn
            )
            for x, y, n, tp, fn in results
        ],
    )


@router.get("/distribution")
def route_prediction_distribution(
    project_hash: uuid.UUID,
    prediction_hash: uuid.UUID,
    domain: PredictionDomain,
    iou: float,
    group: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(100),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine),
) -> metric_query.QueryDistribution:
    # FIXME: extend this to also support data distribution filtered by 'self'
    with Session(engine) as sess:
        tp_fp_dist = None
        if domain != PredictionDomain.FALSE_NEGATIVE:
            tp_fp_dist = metric_query.query_attr_distribution(
                sess=sess,
                tables=TABLES_PREDICTION_TP_FP,
                project_filters={"prediction_hash": [prediction_hash]},
                attr_name=group,
                buckets=buckets,
                filters=filters,
                extra_where=_tp_fp_extra_where(domain, iou, ProjectPredictionAnalytics, prediction_hash),
            )
        fn_dist = None
        if domain in {PredictionDomain.FALSE_NEGATIVE, PredictionDomain.ALL}:
            fn_dist = metric_query.query_attr_distribution(
                sess=sess,
                tables=TABLES_ANNOTATION,
                project_filters={"prediction_hash": [prediction_hash], "project_hash": [project_hash]},
                attr_name=group,
                buckets=buckets,
                filters=filters,
                extra_where=_fn_extra_where(project_hash, prediction_hash),
            )
    if tp_fp_dist is not None and fn_dist is not None:
        group_by = {e.group: e.count for e in fn_dist.results}
        for v in tp_fp_dist.results:
            group_by[v.group] = group_by.setdefault(v.group, 0) + v.count
        return metric_query.QueryDistribution(
            results=[metric_query.QueryDistributionGroup(group=g, count=c) for g, c in group_by.items()]
        )
    elif tp_fp_dist is not None:
        return tp_fp_dist
    elif fn_dist is not None:
        return fn_dist
    else:
        raise RuntimeError("Bug in prediction distribution")


@router.get("/scatter")
def route_prediction_scatter(
    project_hash: uuid.UUID,
    prediction_hash: uuid.UUID,
    domain: PredictionDomain,
    iou: float,
    x_metric: str,
    y_metric: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(10),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine),
) -> metric_query.QueryScatter:
    # FIXME: extend this to also support data distribution filtered by 'self'
    with Session(engine) as sess:
        tp_fp_dist = None
        if domain != PredictionDomain.FALSE_NEGATIVE:
            tp_fp_dist = metric_query.query_attr_scatter(
                sess=sess,
                tables=TABLES_PREDICTION_TP_FP,
                project_filters={"prediction_hash": [prediction_hash]},
                x_metric_name=x_metric,
                y_metric_name=y_metric,
                buckets=buckets,
                filters=filters,
                extra_where=_tp_fp_extra_where(domain, iou, ProjectPredictionAnalytics, prediction_hash),
            )
        fn_dist = None
        if domain in {PredictionDomain.FALSE_NEGATIVE, PredictionDomain.ALL}:
            fn_dist = metric_query.query_attr_scatter(
                sess=sess,
                tables=TABLES_ANNOTATION,
                project_filters={"prediction_hash": [prediction_hash], "project_hash": [project_hash]},
                x_metric_name=x_metric,
                y_metric_name=y_metric,
                buckets=buckets,
                filters=filters,
                extra_where=_fn_extra_where(project_hash, prediction_hash),
            )
    if tp_fp_dist is not None and fn_dist is not None:
        group_by: Dict[Tuple[float, float], int] = {(e.x, e.y): e.n for e in fn_dist.samples}
        for v in tp_fp_dist.samples:
            group_by[v.x, v.y] = group_by.setdefault((v.x, v.y), 0) + v.n
        return metric_query.QueryScatter(
            results=[metric_query.QueryScatterPoint(x=x, y=y, n=n) for (x, y), n in group_by.items()]
        )
    elif tp_fp_dist is not None:
        return tp_fp_dist
    elif fn_dist is not None:
        return fn_dist
    else:
        raise RuntimeError("Bug in prediction distribution")


@router.get("/search")
def route_prediction_search(
    project_hash: uuid.UUID,
    prediction_hash: uuid.UUID,
    domain: PredictionDomain,
    iou: float,
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    order_by: Optional[str] = None,
    desc: bool = False,
    offset: int = Query(0, ge=0),
    limit: int = Query(1000, le=1000),
    engine: Engine = Depends(dep_engine),
) -> AnalysisSearch:
    # FIXME: clean up the implementation with more shared logic.
    # Where conditions for FN table
    fn_select = None
    if domain in {PredictionDomain.ALL, PredictionDomain.FALSE_NEGATIVE}:
        fn_base_table = TABLES_ANNOTATION.primary
        fn_where = search_query.search_filters(
            tables=TABLES_ANNOTATION,
            base=fn_base_table.analytics,
            search=filters,
            project_filters={"project_hash": [project_hash], "prediction_hash": [prediction_hash]},
        ) + [
            ProjectPredictionAnalyticsFalseNegatives.project_hash == project_hash,
            ProjectPredictionAnalyticsFalseNegatives.prediction_hash == prediction_hash,
            ProjectPredictionAnalyticsFalseNegatives.du_hash == ProjectAnnotationAnalytics.du_hash,
            ProjectPredictionAnalyticsFalseNegatives.frame == ProjectAnnotationAnalytics.frame,
            ProjectPredictionAnalyticsFalseNegatives.annotation_hash == ProjectAnnotationAnalytics.annotation_hash,
        ]
        fn_order_attr = literal(1)
        if order_by is not None and order_by in fn_base_table.metrics or order_by in fn_base_table.enums:
            fn_order_attr = getattr(fn_base_table.analytics, order_by)
        fn_select = (
            select(
                ProjectPredictionAnalyticsFalseNegatives.du_hash,
                ProjectPredictionAnalyticsFalseNegatives.frame,
                ProjectPredictionAnalyticsFalseNegatives.annotation_hash,
                2,
                *([fn_order_attr.label("order_by")] if domain == PredictionDomain.ALL else []),
            )
            .where(*fn_where)
            .order_by(fn_order_attr.desc() if desc else fn_order_attr)
        )

    # Where conditions for TP table
    p_select = None
    if domain != PredictionDomain.FALSE_NEGATIVE:
        p_base_table = TABLES_PREDICTION_TP_FP.annotation
        p_where = search_query.search_filters(
            tables=TABLES_PREDICTION_TP_FP,
            base=p_base_table.analytics,
            search=filters,
            project_filters={"project_hash": [project_hash], "prediction_hash": [prediction_hash]},
        )
        tp_cond = (ProjectPredictionAnalytics.iou >= iou) & (ProjectPredictionAnalytics.match_duplicate_iou < iou)
        tp_cond_select = tp_cond.cast(Integer)
        if domain == PredictionDomain.TRUE_POSITIVE:
            p_where.append(tp_cond)
            tp_cond_select = literal(1)
        elif domain == PredictionDomain.FALSE_POSITIVE:
            p_where.append(~tp_cond)
            tp_cond_select = literal(0)
        p_order_attr = literal(1)
        if order_by is not None and order_by in p_base_table.metrics or order_by in p_base_table.enums:
            p_order_attr = getattr(p_base_table.analytics, order_by)
        p_select = (
            select(  # type: ignore
                ProjectPredictionAnalytics.du_hash,
                ProjectPredictionAnalytics.frame,
                ProjectPredictionAnalytics.annotation_hash,
                tp_cond_select,
                *([p_order_attr.label("order_by")] if domain == PredictionDomain.ALL else []),
            )
            .where(*p_where)
            .order_by(p_order_attr.desc() if desc else p_order_attr)
        )

    # Conditionally union to 1 select query.
    if fn_select is not None and p_select is not None:
        f_select = fn_select.union_all(p_select).order_by("order_by")
    elif fn_select is not None:
        f_select = fn_select
    elif p_select is not None:
        f_select = p_select
    else:
        raise RuntimeError("Impossible prediction domain case")

    # Apply ordering
    if domain == PredictionDomain.ALL:
        f_select = f_select.order_by(desc_fn("order_by") if desc else "order_by")

    # Offset & Limit
    f_select = f_select.offset(offset).limit(limit + 1)

    with Session(engine) as sess:
        print(f"DEBUG PREDICTION SEARCH: {f_select}")
        search_results = sess.exec(f_select).fetchall()

    ty_lookup = {0: "FP", 1: "TP", 2: "FN"}

    return AnalysisSearch(
        truncated=len(search_results) == limit + 1,
        results=[
            f"{du_hash}_{frame}_{annotation_hash}_{ty_lookup[ty]}"
            for du_hash, frame, annotation_hash, ty, *rest in search_results[:-1]
        ],
    )
