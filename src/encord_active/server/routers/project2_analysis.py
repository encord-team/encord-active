import functools
import math
import uuid
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from scipy.stats import ks_2samp
from sqlalchemy import func, Numeric, Integer
from sqlalchemy.sql.operators import is_not
from sqlmodel import Session, select

from encord_active.server.dependencies import DataOrAnnotateItem, parse_data_or_annotate_item
from encord_active.server.routers.project2_engine import engine
from encord_active.server.routers.queries import (
    metric_query,
    search_query,
    similarity_query,
)
from encord_active.server.routers.queries.domain_query import (
    TABLES_ANNOTATION,
    TABLES_DATA,
    DomainTables,
    Tables,
)
from encord_active.server.routers.queries.metric_query import literal_bucket_depends
from encord_active.server.routers.queries.search_query import (
    SearchFiltersFastAPIDepends,
)

router = APIRouter(
    prefix="/{project_hash}/analysis/{domain}",
)

MODERATE_IQR_SCALE = 1.5
SEVERE_IQR_SCALE = 2.5


class AnalysisDomain(Enum):
    Data = "data"
    Annotation = "annotation"


def _get_metric_domain_tables(domain: AnalysisDomain) -> Tables:
    if domain == AnalysisDomain.Data:
        return TABLES_DATA
    else:
        return TABLES_ANNOTATION


def _pack_id(du_hash: uuid.UUID, frame: int, annotation_hash: Optional[str] = None) -> str:
    if annotation_hash is None:
        return f"{du_hash}_{frame}"
    else:
        return f"{du_hash}_{frame}_{annotation_hash}"


def _unpack_id(ident: str) -> Tuple[uuid.UUID, int, Optional[str]]:
    values = ident.split("_")
    if len(values) == 2:
        du_hash_str, frame_str = values
        annotation_hash = None
    else:
        du_hash_str, frame_str, annotation_hash = values
    return uuid.UUID(du_hash_str), int(frame_str), annotation_hash


@router.get("/summary")
def metric_summary(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
) -> metric_query.QuerySummary:
    tables = _get_metric_domain_tables(domain)
    with Session(engine) as sess:
        return metric_query.query_attr_summary(
            sess=sess,
            tables=tables,
            project_filters={
                "project_hash": [project_hash],
            },
            filters=filters,
        )


class AnalysisSearch(BaseModel):
    truncated: bool
    results: List[str]


@router.get("/search")
def metric_search(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    order_by: Optional[str] = None,
    desc: bool = False,
    offset: int = 0,
    limit: int = 1000,
) -> AnalysisSearch:
    tables = _get_metric_domain_tables(domain)
    base_table = tables.primary
    where = search_query.search_filters(
        tables=tables,
        base=base_table.analytics,
        search=filters,
        project_filters={"project_hash": [project_hash]},
    )
    print(f"Filter debugging =>: {filters}")
    print(f"Where debugging =>: {where}")

    with Session(engine) as sess:
        query = select(*[getattr(base_table.analytics, join_attr) for join_attr in base_table.join]).where(*where)

        if order_by is not None:
            if order_by in base_table.metrics or order_by in base_table.enums:
                order_by_attr = getattr(base_table.analytics, order_by)
                if desc:
                    order_by_attr.desc()
                query = query.order_by(order_by_attr)

        # + 1 to detect truncation.
        query = query.offset(offset).limit(limit + 1)
        print(f"DEBUGGING: {query}")
        search_results = sess.exec(query).fetchall()

    truncated = len(search_results) == limit + 1

    return AnalysisSearch(
        truncated=truncated,
        results=[
            _pack_id(**{str(join_attr): value for join_attr, value in zip(base_table.join, result)})
            for result in search_results[:-1]
        ],
    )


@router.get("/scatter")
def scatter_2d_data_metric(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    x_metric: str,
    y_metric: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(1000),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
) -> metric_query.QueryScatter:
    tables = _get_metric_domain_tables(domain)
    with Session(engine) as sess:
        return metric_query.query_attr_scatter(
            sess=sess,
            tables=tables,
            project_filters={"project_hash": [project_hash]},
            x_metric_name=x_metric,
            y_metric_name=y_metric,
            buckets=buckets,
            filters=filters,
        )


@router.get("/distribution")
def get_metric_distribution(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    group: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(100),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
) -> metric_query.QueryDistribution:
    tables = _get_metric_domain_tables(domain)
    with Session(engine) as sess:
        return metric_query.query_attr_distribution(
            sess=sess,
            tables=tables,
            project_filters={
                "project_hash": [project_hash],
            },
            attr_name=group,
            buckets=buckets,
            filters=filters,
        )


class Query2DEmbedding(BaseModel):
    count: int
    reductions: List[metric_query.QueryScatterPoint]


@router.get("/reductions/{reduction_hash}/summary")
def get_2d_embedding_summary(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    reduction_hash: uuid.UUID,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(10),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
) -> Query2DEmbedding:
    tables = _get_metric_domain_tables(domain)
    domain_tables = tables.primary
    # FIXME: reduction_hash search filters will not work as they are on the wrong reduction_hash.
    #  Will work correctly IFF reduction hash filtering only happens on the same reduction hash
    where = search_query.search_filters(
        tables=tables,
        base=domain_tables.reduction,
        search=filters,
        project_filters={
            "project_hash": [project_hash],
        },
    )
    with Session(engine) as sess:
        # FIXME: support variable bucketing.
        round_digits = None if buckets is None else int(math.log10(buckets))
        if engine.dialect == "sqlite":
            x_attr = func.round(domain_tables.reduction.x, round_digits)
            y_attr = func.round(domain_tables.reduction.y, round_digits)
        else:
            x_attr = domain_tables.reduction.x.cast(Numeric(6 + round_digits, round_digits))
            y_attr = domain_tables.reduction.y.cast(Numeric(6 + round_digits, round_digits))
        query = (
            select(  # type: ignore
                x_attr.label("xv"),
                y_attr.label("yv"),
                func.count(),
            )
            .where(
                domain_tables.reduction.reduction_hash == reduction_hash,
                *where,
            )
            .group_by(
                "xv",
                "yv",
            )
        )
        print(f"DEBUGGING REDUCTION: {query}")
        results = sess.exec(query).fetchall()
        print(f"DEBUG SMORE: {results}")
    return Query2DEmbedding(
        count=sum(n for x, y, n in results),
        reductions=[metric_query.QueryScatterPoint(x=x, y=y, n=n) for x, y, n in results],
    )


@functools.lru_cache(maxsize=2)
def _get_similarity_query(project_hash: uuid.UUID, domain: AnalysisDomain) -> similarity_query.SimilarityQuery:
    tables = _get_metric_domain_tables(domain)
    base_domain: DomainTables = tables.primary
    with Session(engine) as sess:
        query = select(
            base_domain.metadata.embedding_clip,
            *base_domain.select_args(base_domain.metadata)
        ).where(
            # FIXME: will break for nearest embedding on predictions
            base_domain.metadata.project_hash == project_hash,  # type: ignore
            is_not(base_domain.metadata.embedding_clip, None),
        )
        results = sess.exec(query).fetchall()
    embeddings = [np.frombuffer(e[0], dtype=np.float32) for e in results]
    return similarity_query.SimilarityQuery(embeddings, results)


@router.get("/similarity/{item}")
def search_similarity(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    embedding: Literal["embedding_clip"],
    similarity_item: DataOrAnnotateItem = Depends(parse_data_or_annotate_item),
) -> List[similarity_query.SimilarityResult]:
    tables = _get_metric_domain_tables(domain)
    limit = 50
    base_domain = tables.primary
    if embedding != "embedding_clip":
        raise ValueError("Unsupported embedding")
    join_attr_set = {
        "du_hash": similarity_item.du_hash,
        "frame": similarity_item.frame,
    }
    if similarity_item.annotation_hash is not None:
        join_attr_set["annotation_hash"] = similarity_item.annotation_hash

    with Session(engine) as sess:
        src_embedding = sess.exec(
            select(
                base_domain.metadata.embedding_clip
            ).where(
                base_domain.metadata.project_hash == project_hash,
                *[
                    getattr(base_domain.metadata, join_attr) == join_attr_set[join_attr]
                    for join_attr in base_domain.join
                ],
            )
        ).first()
        if src_embedding is None:
            raise ValueError("Source entry does not exist or missing embedding")

        if engine.dialect.name == 'postgresql':
            pg_query = select(
                base_domain.metadata.embedding_clip.l2_distance(src_embedding).label("similarity"),  # type: ignore
                *base_domain.select_args(base_domain.metadata),
            ).where(
                base_domain.metadata.project_hash == project_hash,
                functools.reduce(lambda a, b: a | b, [
                    getattr(base_domain.metadata, join_attr) != join_attr_set[join_attr]
                    for join_attr in base_domain.join
                ])
            ).order_by(
                "similarity"
            ).limit(limit)
            pg_results = sess.exec(pg_query).fetchall()
            return [
                similarity_query.pack_similarity_result(tuple(similarity_item), similarity)
                for similarity, *similarity_item in pg_results
            ]

    # Return via fallback methods (sqlite & similar)
    similarity_query_impl = _get_similarity_query(project_hash, domain)
    return similarity_query_impl.query(np.frombuffer(src_embedding, dtype=np.float32), k=limit)


class MetricDissimilarityResult(BaseModel):
    results: Dict[str, float]


@router.get("/project_compare/metric_dissimilarity")
def compare_metric_dissimilarity(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    compare_project_hash: uuid.UUID,
) -> MetricDissimilarityResult:
    tables = _get_metric_domain_tables(domain)
    base_domain = tables.primary
    dissimilarity = {}
    with Session(engine) as sess:
        for metric_name in base_domain.metrics.keys():
            metric_attr: float = getattr(base_domain.analytics, metric_name)
            all_data_1 = sess.exec(
                select(
                    metric_attr,
                ).where(base_domain.analytics.project_hash == project_hash, is_not(metric_attr, None))
            ).fetchall()
            all_data_2 = sess.exec(
                select(
                    metric_attr,
                ).where(base_domain.analytics.project_hash == compare_project_hash, is_not(metric_attr, None))
            ).fetchall()
            if len(all_data_1) > 0 and len(all_data_2) > 0:
                k_score, _ = ks_2samp(np.array(all_data_1), np.array(all_data_2))
                dissimilarity[metric_name] = k_score

    return MetricDissimilarityResult(
        results=dissimilarity,
    )
