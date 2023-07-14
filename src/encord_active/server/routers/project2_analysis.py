import functools
import json
import math
import uuid
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from fastapi import APIRouter
from pynndescent import NNDescent
from scipy.stats import ks_2samp
from sqlalchemy import func
from sqlalchemy.sql.operators import between_op, in_op, is_not, not_between_op
from sqlmodel import Session, select

from encord_active.db.metrics import (
    AnnotationMetrics,
    DataMetrics,
    MetricDefinition,
)
from encord_active.db.models import (
    AnnotationType,
    ProjectAnnotationAnalytics,
    ProjectDataAnalytics, ProjectDataAnalyticsExtra, ProjectAnnotationAnalyticsExtra, ProjectDataAnalyticsReduced,
    ProjectAnnotationAnalyticsReduced,
)
from encord_active.server.routers.project2_engine import engine
from encord_active.server.routers.queries import metric_query
from encord_active.server.routers.queries.domain_query import Tables, TABLES_DATA, TABLES_ANNOTATION, SearchFilters

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

def _get_metric_domain(
    domain: AnalysisDomain,
) -> Tuple[
    Union[Type[ProjectDataAnalytics], Type[ProjectAnnotationAnalytics]],
    Dict[str, MetricDefinition],
    Optional[str],
    Dict[str, dict],
    Union[Type[ProjectDataAnalyticsExtra], Type[ProjectAnnotationAnalyticsExtra]],
    Union[Type[ProjectDataAnalyticsReduced], Type[ProjectAnnotationAnalyticsReduced]],
]:
    if domain == AnalysisDomain.Data:
        return ProjectDataAnalytics, DataMetrics, None, {}, ProjectDataAnalyticsExtra, ProjectDataAnalyticsReduced
    elif domain == AnalysisDomain.Annotation:
        enum_props: Dict[str, dict] = {
            "feature_hash": {"type": "ontology"},
            "annotation_type": {
                "type": "set",
                "values": {annotation_type.value: annotation_type.name for annotation_type in AnnotationType},
            },
        }
        return ProjectAnnotationAnalytics, AnnotationMetrics, "object_hash", enum_props, \
            ProjectAnnotationAnalyticsExtra, ProjectAnnotationAnalyticsReduced
    else:
        raise ValueError(f"Bad domain: {domain}")


def _load_metric(
    sess: Session,
    project_hash: uuid.UUID,
    metric_name: str,
    metric: MetricDefinition,
    cls: Type[Union[ProjectDataAnalytics, ProjectAnnotationAnalytics]],
    iqr_only: bool = False,
) -> Optional[dict]:
    metric_attr = getattr(cls, metric_name)
    where = [cls.project_hash == project_hash, is_not(metric_attr, None)]
    count: int = sess.exec(select(func.count()).where(*where)).first() or 0  # type: ignore
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
        metric_min = sess.exec(select(func.min(metric_attr)).where(*where)).first()  # type: ignore
        metric_max = sess.exec(select(func.max(metric_attr)).where(*where)).first()  # type: ignore
        median = sess.exec(
            select(metric_attr).where(*where).order_by(metric_attr).offset(median_offset).limit(1)
        ).first()  # type: ignore
    q1 = sess.exec(select(metric_attr).where(*where).order_by(metric_attr).offset(q1_offset).limit(1)).first() or 0
    q3 = sess.exec(select(metric_attr).where(*where).order_by(metric_attr).offset(q3_offset).limit(1)).first() or 0

    # Calculate count of moderate & severe outliers
    if iqr_only:
        severe_count = 0
        moderate_count = 0
    else:
        iqr = q3 - q1
        moderate_lb, moderate_ub = q1 - MODERATE_IQR_SCALE * iqr, q3 + MODERATE_IQR_SCALE * iqr
        severe_lb, severe_ub = q1 - SEVERE_IQR_SCALE * iqr, q3 + SEVERE_IQR_SCALE * iqr
        severe_count = sess.exec(
            select(func.count()).where(*where, not_between_op(metric_attr, severe_lb, severe_ub))  # type: ignore
        ).first()
        moderate_count = sess.exec(
            select(func.count()).where(  # type: ignore
                *where,
                not_between_op(metric_attr, moderate_lb, moderate_ub),
                between_op(metric_attr, severe_lb, severe_ub),
            )
        ).first()

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


@router.get("/summary")
def metric_summary(project_hash: uuid.UUID, domain: AnalysisDomain):
    domain_ty, domain_metrics, extra_key, domain_enums, domain_ty_extra, domain_ty_2d = _get_metric_domain(domain)
    with Session(engine) as sess:
        count: int = sess.exec(select(func.count()).where(domain_ty.project_hash == project_hash)).first()  # type: ignore
    metrics = {
        metric_name: _load_metric(sess, project_hash, metric_name, metric, domain_ty)
        for metric_name, metric in domain_metrics.items()
    }
    return {
        "count": count,
        "metrics": {k: v for k, v in metrics.items() if v is not None},
        "enums": domain_enums,
    }


@router.get("/search")
def metric_search(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    metric_filters: Optional[str] = None,
    metric_outliers: Optional[str] = None,
    enum_filters: Optional[str] = None,
    order_by: Optional[str] = None,
    desc: bool = False,
):
    metric_filters_dict: Optional[Dict[str, Tuple[Union[int, float], Union[int, float]]]] = (
        None if metric_filters is None else json.loads(metric_filters)
    )
    metric_outliers_dict: Optional[Dict[str, Literal["warning", "severe"]]] = (
        None if metric_outliers is None else json.loads(metric_outliers)
    )
    enum_filters_dict: Optional[Dict[str, List[str]]] = None if enum_filters is None else json.loads(enum_filters)
    domain_ty, domain_metrics, domain_grouping, domain_enums, domain_ty_extra, domain_ty_2d = _get_metric_domain(domain)

    # Add metric filtering.
    query_filters = []
    if metric_filters_dict is not None:
        for metric_name, (range_start, range_end) in metric_filters_dict.items():
            metric_meta = domain_metrics[metric_name]
            metric_filter = getattr(domain_ty, metric_name)
            if range_start == range_end:
                query_filters.append(metric_filter == range_start)
            else:
                query_filters.append(metric_filter >= range_start)
                query_filters.append(metric_filter <= range_end)

    if enum_filters_dict is not None:
        for enum_filter_name, enum_filter_list in enum_filters_dict.items():
            if enum_filter_name not in ["feature_hash", "annotation_type"]:
                raise ValueError(f"Unsupported enum filter: {enum_filter_name}")
            enum_filter_col = getattr(domain_ty, enum_filter_name)
            query_filters.append(in_op(enum_filter_col, enum_filter_list))

    with Session(engine) as sess:
        if metric_outliers_dict is not None:
            for metric, outlier_type in metric_outliers_dict.items():
                summary = (
                    _load_metric(sess, project_hash, metric, domain_metrics[raw_metric], domain_ty, iqr_only=True)
                    or {}
                )
                metric_attr = getattr(domain_ty, metric)
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
            domain_ty.du_hash, domain_ty.frame, None if domain_grouping is None else getattr(domain_ty, domain_grouping)
        ).where(domain_ty.project_hash == project_hash, *query_filters)
        if order_by is not None:
            order_by_field = getattr(domain_ty, order_by)
            if desc:
                order_by_field = order_by_field.desc()
            search_query = search_query.order_by(order_by_field)

        search_query = search_query.limit(
            # 1000 max results in a search query (+1 to detect truncation).
            limit=1001
        )
        search_results = sess.exec(search_query).fetchall()
    truncated = len(search_results) == 1001

    return {
        "truncated": truncated,
        "results": [
            {"du_hash": du_hash, "frame": frame, domain_grouping: group_hash}
            for du_hash, frame, group_hash in search_results[:-1]
        ]
        if domain_grouping is not None
        else [{"du_hash": du_hash, "frame": frame} for du_hash, frame, group_hash in search_results[:-1]],
    }


@router.get("/scatter")
def scatter_2d_data_metric(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    x_metric: str,
    y_metric: str,
    buckets: Literal[10, 100, 1000] = 10,
    filters: Optional[SearchFilters] = None,
):
    tables = _get_metric_domain_tables(domain)
    domain_tables = tables.annotation or tables.data
    with Session(engine) as sess:
        return metric_query.query_attr_scatter(
            sess=sess,
            tables=tables,
            extra=[],  # FIXME: define extra for scatter plot visualization
            where=[
                # FIXME: make project_hash filter easier to define (hardcode!).
                domain_tables.analytics.project_hash == project_hash
            ],
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
    buckets: Literal[10, 100, 1000] = 10,
    filters: Optional[SearchFilters] = None,
):
    tables = _get_metric_domain_tables(domain)
    domain_tables = tables.annotation or tables.data
    with Session(engine) as sess:
        return metric_query.query_attr_distribution(
            sess=sess,
            tables=tables,
            extra=[],
            where=[
                domain_tables.analytics.project_hash == project_hash
            ],
            attr_name=group,
            buckets=buckets,
            filters=filters,
        )


@router.get("/2d_embeddings/{reduction_hash}/summary")
def get_2d_embedding_summary(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    reduction_hash: uuid.UUID,
    buckets: Literal[10, 100, 1000] = 10,
    filters: Optional[SearchFilters] = None,
):
    tables = _get_metric_domain_tables(domain)
    domain_tables = tables.annotation or tables.data

    with Session(engine) as sess:
        round_digits = None if buckets is None else int(math.log10(buckets))
        query = select(
            func.round(domain_tables.reduction.x, round_digits),
            func.round(domain_tables.reduction.y, round_digits),
            func.count(),
        ).where(
            domain_tables.reduction.project_hash == project_hash,
            domain_tables.reduction.reduction_hash == reduction_hash,
            # FIXME:
        ).group_by(
            func.round(domain_tables.reduction.x, round_digits),
            func.round(domain_tables.reduction.y, round_digits),
        )
        results = sess.exec(query)
    return {
        "count": sum(n for x, y, n in results),
        "2d_embedding": [
            {"x": x, "y": y, "n": n}
            for x, y, n in results
        ]
    }


@functools.lru_cache(maxsize=2)
def _get_nn_descent(
    project_hash: uuid.UUID, domain: AnalysisDomain
) -> Tuple[NNDescent, List[Tuple[Optional[bytes], uuid.UUID, int, Optional[str]]]]:
    domain_ty, domain_metrics, object_key, domain_enums, domain_ty_extra, domain_ty_2d = _get_metric_domain(domain)
    with Session(engine) as sess:
        query = select(
            domain_ty_extra.embedding_clip,
            domain_ty_extra.du_hash,
            domain_ty_extra.frame,
            *([] if object_key is None else [getattr(domain_ty_extra, object_key)]),
        ).where(domain_ty_extra.project_hash == project_hash, is_not(domain_ty_extra.embedding_clip, None))
        results = sess.exec(query).fetchall()
    embeddings = np.stack([np.frombuffer(e[0], dtype=np.float) for e in results]).astype(np.float32)  # type: ignore
    index = NNDescent(embeddings, n_neighbors=50, metric="cosine")
    return index, results  # type: ignore


@router.get("/similarity/{du_hash}/{frame}/{object_hash}")
@router.get("/similarity/{du_hash}/{frame}/")
def search_similarity(
        project_hash: uuid.UUID, domain: AnalysisDomain, du_hash: uuid.UUID, frame: int,
        embedding: str, object_hash: Optional[str] = None
):
    domain_ty, domain_metrics, object_key, domain_enums, domain_ty_extra, domain_ty_2d = _get_metric_domain(domain)
    if embedding != "embedding_clip":
        raise ValueError("Unsupported embedding")
    with Session(engine) as sess:
        src_embedding = sess.exec(
            select(domain_ty_extra.embedding_clip).where(
                domain_ty_extra.project_hash == project_hash,
                domain_ty_extra.du_hash == du_hash,
                domain_ty_extra.frame == frame,
                *([] if object_key is None else (getattr(domain_ty_extra, object_key) == object_hash))
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
    # FIXME: try and convert to sql query for efficiency.
    domain_ty, domain_metrics, extra_key, domain_enums, domain_ty_extra, domain_ty_2d = _get_metric_domain(domain)
    dissimilarity = {}
    with Session(engine) as sess:
        for metric_name in domain_metrics:
            metric_attr: float = getattr(domain_ty, metric_name)
            all_data_1 = sess.exec(select(
                metric_attr,
            ).where(
                domain_ty.project_hash == project_hash,
                is_not(metric_attr, None)
            )).fetchall()
            all_data_2 = sess.exec(select(
                metric_attr,
            ).where(
                domain_ty.project_hash == compare_project_hash,
                is_not(metric_attr, None)
            )).fetchall()
            if len(all_data_1) > 0 and len(all_data_2) > 0:
                k_score, _ = ks_2samp(np.array(all_data_1), np.array(all_data_2))
                dissimilarity[metric_name] = k_score

    return {
        "results": dissimilarity,
    }
