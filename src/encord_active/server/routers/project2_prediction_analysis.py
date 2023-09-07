import uuid
from enum import Enum
from typing import Literal, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import Integer, Float, desc as desc_fn
from sqlmodel import Session
from sqlalchemy.engine import Engine

from encord_active.db.models import (
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsFalseNegatives,
    ProjectAnnotationAnalytics,
)
from encord_active.server.dependencies import dep_engine
from encord_active.server.routers.project2_analysis import AnalysisSearch
from encord_active.server.routers.queries import search_query, metric_query
from encord_active.server.routers.queries.domain_query import Tables, TABLES_ANNOTATION, TABLES_PREDICTION_TP_FP
from encord_active.server.routers.queries.metric_query import literal_bucket_depends
from encord_active.server.routers.queries.search_query import SearchFiltersFastAPIDepends


class PredictionDomain(Enum):
    ALL = "a"
    POSITIVE = "p"
    TRUE_POSITIVE = "tp"
    FALSE_POSITIVE = "fp"
    FALSE_NEGATIVE = "fn"


def _load_tables(domain: PredictionDomain) -> Tables:
    if domain == PredictionDomain.FALSE_NEGATIVE:
        pass

    raise ValueError(f"Unsupported domain: {domain}")


router = APIRouter(
    prefix="/analytics/{domain}",
)


class PredictionQueryScatterPoint(BaseModel):
    x: float
    y: float
    n: int
    tp: float


class PredictionQuery2DEmbedding(BaseModel):
    count: int
    reductions: List[PredictionQueryScatterPoint]


@router.get("/reductions/{reduction_hash}/summary")
def get_2d_embedding_summary_prediction(
    project_hash: uuid.UUID,
    prediction_hash: uuid.UUID,
    domain: PredictionDomain,
    reduction_hash: uuid.UUID,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(10),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine),
) -> PredictionQuery2DEmbedding:
    return PredictionQuery2DEmbedding(
        count=0,
        reductions=[],
    )


@router.get("/distribution")
def prediction_metric_distribution(
    project_hash: uuid.UUID,
    prediction_hash: uuid.UUID,
    domain: PredictionDomain,
    group: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(100),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine),
) -> metric_query.QueryDistribution:
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


@router.get("/scatter")
def prediction_metric_scatter(
    prediction_hash: uuid.UUID,
    domain: PredictionDomain,
    x_metric: str,
    y_metric: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(10),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine),
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


@router.get("/search")
def prediction_search(
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
    # Where conditions for FN table
    fn_select = None
    if domain in {PredictionDomain.ALL, PredictionDomain.FALSE_NEGATIVE}:
        fn_where = search_query.search_filters(
            tables=TABLES_ANNOTATION,
            base=TABLES_ANNOTATION.analytics,
            search=filters,
            project_filters={"project_hash": [project_hash], "prediction_hash": [prediction_hash]},
        ) + [
            ProjectPredictionAnalyticsFalseNegatives.project_hash == project_hash,
            ProjectPredictionAnalyticsFalseNegatives.prediction_hash == prediction_hash,
            ProjectPredictionAnalyticsFalseNegatives.du_hash == ProjectAnnotationAnalytics.du_hash,
            ProjectPredictionAnalytics,
        ]

    # Where conditions for TP table
    p_select = None
    if domain != PredictionDomain.FALSE_NEGATIVE:
        p_where = search_query.search_filters(
            tables=TABLES_PREDICTION_TP_FP,
            base=TABLES_PREDICTION_TP_FP.analytics,
            search=filters,
            project_filters={"project_hash": [project_hash], "prediction_hash": [prediction_hash]},
        )
        tp_cond = (
            ((ProjectPredictionAnalytics.iou >= iou) & (ProjectPredictionAnalytics.match_duplicate_iou < iou))
            .cast(Integer)
            .cast(Float)
        )
        if domain == PredictionDomain.TRUE_POSITIVE:
            p_where.append(tp_cond)
        elif domain == PredictionDomain.FALSE_POSITIVE:
            p_where.append(~tp_cond)

    # Conditionally union to 1 select query.
    if fn_select is not None and p_select is not None:
        pass  # FIXME: union all
    elif fn_select is not None:
        f_select = fn_select
    elif p_select is not None:
        f_select = p_select
    else:
        raise RuntimeError("Impossible prediction domain case")

    # Apply ordering
    if desc:
        f_select = f_select.order_by(desc_fn("order_k"))
    else:
        f_select = f_select.order_by("order_k")
    f_select = f_select.offset(offset).limit(limit + 1)

    with Session(engine) as sess:
        results = sess.exec(f_select).fetchall()

    return AnalysisSearch(
        truncated=True,
        results=[],
    )
