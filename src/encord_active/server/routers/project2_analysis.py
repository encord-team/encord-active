import functools
import math
import uuid
from enum import Enum
from typing import List, Literal, Optional, Tuple

import numpy as np
from fastapi import APIRouter
from pynndescent import NNDescent
from scipy.stats import ks_2samp
from sqlalchemy import func
from sqlalchemy.sql.operators import is_not
from sqlmodel import Session, select

from encord_active.server.routers.project2_engine import engine
from encord_active.server.routers.queries import metric_query, search_query
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


@router.get("/summary")
def metric_summary(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
):
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


@router.get("/search")
def metric_search(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    order_by: Optional[str] = None,
    desc: bool = False,
    offset: int = 0,
    limit: int = 1000,
):
    tables = _get_metric_domain_tables(domain)
    base_table = tables.annotation or tables.data
    where = search_query.search_filters(
        tables=tables,
        base="analytics",  # FIXME: this should select the best base table (reduction if no analytics filters)
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

    return {
        "truncated": truncated,
        "results": [
            {join_attr: value for join_attr, value in zip(base_table.join, result)} for result in search_results[:-1]
        ],
    }


@router.get("/scatter")
def scatter_2d_data_metric(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    x_metric: str,
    y_metric: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(1000),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
):
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
):
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


@router.get("/2d_embeddings/{reduction_hash}/summary")
def get_2d_embedding_summary(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    reduction_hash: uuid.UUID,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(10),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
):
    tables = _get_metric_domain_tables(domain)
    domain_tables = tables.annotation or tables.data
    # FIXME: reduction_hash search filters will not work as they are on the wrong reduction_hash.
    #  Will work correctly IFF reduction hash filtering only happens on the same reduction hash
    where = search_query.search_filters(
        tables=tables,
        base="reduction",
        search=filters,
        project_filters={
            "project_hash": [project_hash],
        },
    )
    with Session(engine) as sess:
        round_digits = None if buckets is None else int(math.log10(buckets))
        query = (
            select(  # type: ignore
                func.round(domain_tables.reduction.x, round_digits),
                func.round(domain_tables.reduction.y, round_digits),
                func.count(),
            )
            .where(
                domain_tables.reduction.reduction_hash == reduction_hash,
                *where,
            )
            .group_by(
                func.round(domain_tables.reduction.x, round_digits),
                func.round(domain_tables.reduction.y, round_digits),
            )
        )
        results = sess.exec(query)
    return {"count": sum(n for x, y, n in results), "2d_embedding": [{"x": x, "y": y, "n": n} for x, y, n in results]}


@functools.lru_cache(maxsize=2)
def _get_nn_descent(
    project_hash: uuid.UUID, domain: AnalysisDomain
) -> Tuple[NNDescent, List[Tuple[Optional[bytes], uuid.UUID, int, Optional[str]]]]:
    tables = _get_metric_domain_tables(domain)
    base_domain: DomainTables = tables.annotation or tables.data
    with Session(engine) as sess:
        query = select(
            base_domain.metadata.embedding_clip,
            *[getattr(base_domain.metadata, join_attr) for join_attr in base_domain.join],
        ).where(
            # FIXME: will break for nearest embedding on predictions
            base_domain.metadata.project_hash == project_hash,  # type: ignore
            is_not(base_domain.metadata.embedding_clip, None),
        )
        results = sess.exec(query).fetchall()
    embeddings = np.stack([np.frombuffer(e[0], dtype=np.float) for e in results]).astype(np.float32)  # type: ignore
    index = NNDescent(embeddings, n_neighbors=50, metric="cosine")
    return index, results  # type: ignore


@router.get("/similarity/{du_hash}/{frame}/{object_hash}")
@router.get("/similarity/{du_hash}/{frame}/")
def search_similarity(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    du_hash: uuid.UUID,
    frame: int,
    embedding: str,
    object_hash: Optional[str] = None,
):
    tables = _get_metric_domain_tables(domain)
    base_domain = tables.annotation or tables.data
    if embedding != "embedding_clip":
        raise ValueError("Unsupported embedding")
    join_attr_set = {
        "du_hash": du_hash,
        "frame": frame,
    }
    if object_hash is not None:
        join_attr_set["object_hash"] = object_hash
    with Session(engine) as sess:
        src_embedding = sess.exec(
            select(base_domain.metadata.embedding_clip).where(
                # FIXME: prediction needs separate join!
                base_domain.metadata.project_hash == project_hash,  # type: ignore
                *[
                    getattr(base_domain.metadata, join_attr) == join_attr_set[join_attr]
                    for join_attr in base_domain.join
                ],
            )
        ).first()
        if src_embedding is None:
            raise ValueError("Source entry does not exist or missing embedding")

    index, results = _get_nn_descent(project_hash, domain)
    indices, similarity = index.query(np.frombuffer(src_embedding, dtype=np.float).reshape(1, -1), k=50)  # type: ignore
    seen = set()
    similarity_results = []
    for i, s in zip(indices[0], similarity[0]):
        if i in seen:
            continue
        _, s_du_hash, s_frame, *keys = results[i]
        if s_du_hash == du_hash and s_frame == frame:  # FIXME: object_hash comparison
            seen.add(i)  # Do not return 'self'
            continue
        similarity_results.append(
            {
                "du_hash": s_du_hash,
                "frame": s_frame,
                "similarity": s,
                **({} if len(keys) == 0 else {"object_hash": keys[0]}),
            }
        )
        seen.add(i)
    return {"results": similarity_results}


@router.get("/project_compare/metric_dissimilarity")
def compare_metric_dissimilarity(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    compare_project_hash: uuid.UUID,
):
    tables = _get_metric_domain_tables(domain)
    base_domain = tables.annotation or tables.data
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

    return {
        "results": dissimilarity,
    }
