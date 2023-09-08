import dataclasses
import math
from enum import IntEnum
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from fastapi import Depends, Query
from pydantic import BaseModel
from sqlalchemy import Numeric
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.functions import count as sql_count_raw
from sqlalchemy.sql.functions import max as sql_max_raw
from sqlalchemy.sql.functions import min as sql_min_raw
from sqlalchemy.sql.functions import sum as sql_sum_raw
from sqlalchemy.sql.operators import between_op, is_not, not_between_op
from sqlmodel import Session, SQLModel, func, select
from sqlmodel.sql.expression import Select

from encord_active.db.enums import EnumDefinition
from encord_active.db.metrics import MetricDefinition, MetricType
from encord_active.server.routers.queries import search_query
from encord_active.server.routers.queries.domain_query import (
    AnalyticsTable,
    ProjectFilters,
    Tables,
)

# Type hints
TType = TypeVar("TType")
sql_count: Callable[[], int] = sql_count_raw  # type: ignore
sql_min: Callable[[TType], TType] = sql_min_raw  # type: ignore
sql_max: Callable[[TType], TType] = sql_max_raw  # type: ignore
sql_sum: Callable[[TType], TType] = sql_sum_raw  # type: ignore

"""
Severe IQR Scale factor for iqr range
"""
MODERATE_IQR_SCALE = 1.5

"""
Severe IQR Scale factor for iqr range
"""
SEVERE_IQR_SCALE = 2.5


@dataclasses.dataclass
class AttrMetadata:
    group_attr: Union[int, float]
    filter_attr: Union[int, float]
    metric_type: Optional[MetricType]


class AnalysisBuckets(IntEnum):
    B10 = 10
    B100 = 100
    B1000 = 1000


def literal_bucket_depends(default: int) -> Literal[10, 100, 1000]:
    def _parse_buckets(
        buckets: AnalysisBuckets = Query(AnalysisBuckets(default), alias="buckets")
    ) -> Literal[10, 100, 1000]:
        return int(buckets.value)  # type: ignore

    return Depends(_parse_buckets, use_cache=False)


def get_normal_attr_bucket(dialect: Dialect, metric_attr: float, buckets: Optional[Literal[10, 100, 1000]]) -> float:
    round_digits = None if buckets is None else int(math.log10(buckets))
    if dialect.name == "sqlite":
        return metric_attr if round_digits is None else func.ROUND(metric_attr, round_digits)  # type: ignore
    else:
        return metric_attr if round_digits is None else metric_attr.cast(Numeric(1 + round_digits, round_digits))  # type: ignore


def get_float_attr_bucket(dialect: Dialect, metric_attr: float, buckets: Optional[Literal[10, 100, 1000]]) -> float:
    # FIXME: work for different distributions (currently ONLY aspect ratio)
    #  hence we can assume value is near 1.0 (so we currently use same rounding as normal.
    round_digits = None if buckets is None else int(math.log10(buckets))
    return metric_attr if metric_attr is None else func.ROUND(metric_attr.cast(Numeric(20, 10)), round_digits)  # type: ignore


def get_int_attr_bucket(dialect: Dialect, metric_attr: int, buckets: Optional[Literal[10, 100, 1000]]) -> int:
    # FIXME: work for different distributions
    return metric_attr


def get_metric_or_enum(
    dialect: Dialect,
    table: AnalyticsTable,
    attr_name: str,
    metrics: Dict[str, MetricDefinition],
    enums: Dict[str, EnumDefinition],
    buckets: Optional[Literal[10, 100, 1000]] = None,
) -> AttrMetadata:
    metric = metrics.get(attr_name, None)
    if metric is not None:
        raw_metric_attr = getattr(table, attr_name)
        if metric.type == MetricType.NORMAL:
            group_attr = get_normal_attr_bucket(dialect, raw_metric_attr, buckets)
            return AttrMetadata(
                group_attr=group_attr,  # type: ignore
                filter_attr=raw_metric_attr,
                metric_type=metric.type,
            )
        elif metric.type == MetricType.UFLOAT:
            group_attr = get_float_attr_bucket(dialect, raw_metric_attr, buckets)
            return AttrMetadata(
                group_attr=group_attr,  # type: ignore
                filter_attr=raw_metric_attr,
                metric_type=metric.type,
            )
        elif metric.type == MetricType.UINT:
            group_attr = get_int_attr_bucket(dialect, raw_metric_attr, buckets)
            return AttrMetadata(
                group_attr=group_attr,
                filter_attr=raw_metric_attr,
                metric_type=metric.type,
            )
        else:
            raise ValueError("Unknown metric type")

    enum = enums.get(attr_name, None)
    if enum is not None:
        raw_enum_attr = getattr(table, attr_name)
        return AttrMetadata(
            group_attr=raw_enum_attr,
            filter_attr=raw_enum_attr,
            metric_type=None,
        )
    raise ValueError(f"Attribute: {attr_name} is invalid")


class QueryMetricSummary(BaseModel):
    min: float
    q1: float
    median: float
    q3: float
    max: float
    count: int
    moderate: int
    severe: int


def query_metric_attr_summary(
    sess: Session,
    table: Type[SQLModel],
    where: list,
    metric_name: str,
    metrics: Dict[str, MetricDefinition],
) -> QueryMetricSummary:
    metric = metrics[metric_name]
    metric_attr: Union[int, float] = getattr(table, metric_name)
    metric_count, metric_min, metric_max = sess.exec(
        select(sql_count(), sql_min(metric_attr), sql_max(metric_attr)).where(*where, is_not(metric_attr, None))
    ).first() or (0, 0, 0)
    metric_q1 = (
        sess.exec(
            select(metric_attr)
            .where(*where, is_not(metric_attr, None))
            .offset(metric_count // 4)
            .order_by(metric_attr)
            .limit(1)
        ).first()
        or 0
    )
    metric_q2 = (
        sess.exec(
            select(metric_attr)
            .where(*where, is_not(metric_attr, None))
            .offset(metric_count // 2)
            .order_by(metric_attr)
            .limit(1)
        ).first()
        or 0
    )
    metric_q3 = (
        sess.exec(
            select(metric_attr)
            .where(*where, is_not(metric_attr, None))
            .offset((metric_count * 3) // 4)
            .order_by(metric_attr)
            .limit(1)
        ).first()
        or 0
    )

    metric_iqr = metric_q3 - metric_q1
    moderate_lb, moderate_ub = metric_q1 - MODERATE_IQR_SCALE * metric_iqr, metric_q3 + MODERATE_IQR_SCALE * metric_iqr
    severe_lb, severe_ub = metric_q1 - SEVERE_IQR_SCALE * metric_iqr, metric_q3 + SEVERE_IQR_SCALE * metric_iqr
    if metric_name != "metric_random":
        metric_severe = (
            sess.exec(
                select(func.count()).where(*where, not_between_op(metric_attr, severe_lb, severe_ub))  # type: ignore
            ).first()
            or 0
        )
        metric_moderate = (
            sess.exec(
                select(func.count()).where(  # type: ignore
                    *where,
                    not_between_op(metric_attr, moderate_lb, moderate_ub),
                    between_op(metric_attr, severe_lb, severe_ub),
                )
            ).first()
            or 0
        )
    else:
        # metric_random is a special metric that should conceal the presence
        # of outliers, so we override the calculation and always return 0.
        metric_severe = 0
        metric_moderate = 0
    return QueryMetricSummary(
        min=metric_min or 0,
        q1=metric_q1,
        median=metric_q2,
        q3=metric_q3,
        max=metric_max or 0,
        count=metric_count,
        moderate=metric_moderate,
        severe=metric_severe,
    )


class QueryEnumSummary(BaseModel):
    count: int


class QuerySummary(BaseModel):
    count: int
    metrics: Dict[str, QueryMetricSummary]
    enums: Dict[str, QueryEnumSummary]


def query_attr_summary(
    sess: Session,
    tables: Tables,
    project_filters: ProjectFilters,
    filters: Optional[search_query.SearchFilters],
    extra_where: Optional[list] = None,
) -> QuerySummary:
    domain_tables = tables.primary
    where = search_query.search_filters(
        tables=tables,
        base=domain_tables.analytics,
        search=filters,
        project_filters=project_filters,
    ) + (extra_where or [])
    count: int = sess.exec(select(sql_count()).where(*where)).first() or 0
    metrics = {
        metric_name: query_metric_attr_summary(
            sess=sess,
            table=domain_tables.analytics,
            where=where,
            metric_name=metric_name,
            metrics=domain_tables.metrics,
        )
        for metric_name, metric in domain_tables.metrics.items()
    }
    return QuerySummary(
        count=count,
        metrics={k: v for k, v in metrics.items() if v is not None},
        enums={k: QueryEnumSummary(count=count) for k, e in domain_tables.enums.items()},  # FIXME: implement properly
    )


class QueryDistributionGroup(BaseModel):
    group: str
    count: int


class QueryDistribution(BaseModel):
    results: List[QueryDistributionGroup]


def select_for_query_attr_distribution(
    dialect: Dialect,
    tables: Tables,
    project_filters: ProjectFilters,
    attr_name: str,
    buckets: Literal[10, 100, 1000],
    filters: Optional[search_query.SearchFilters],
    extra_where: Optional[list],
) -> "Select[Tuple[Union[int, float], int]]":
    domain_tables = tables.primary
    attr = get_metric_or_enum(
        dialect,
        domain_tables.analytics,
        attr_name,
        domain_tables.metrics,
        domain_tables.enums,
        buckets=buckets,
    )
    where = search_query.search_filters(
        tables=tables,
        base=domain_tables.analytics,
        search=filters,
        project_filters=project_filters,
    )
    return (
        select(
            attr.group_attr.label("g"),  # type: ignore
            sql_count().label("n"),  # type: ignore
        )
        .where(*where, is_not(attr.filter_attr, None), *(extra_where or []))
        .group_by("g")
    )


def query_attr_distribution(
    sess: Session,
    tables: Tables,
    project_filters: ProjectFilters,
    attr_name: str,
    buckets: Literal[10, 100, 1000],
    filters: Optional[search_query.SearchFilters],
    extra_where: Optional[list] = None,
) -> QueryDistribution:
    grouping_query = select_for_query_attr_distribution(
        dialect=sess.bind.dialect,  # type: ignore
        tables=tables,
        project_filters=project_filters,
        attr_name=attr_name,
        buckets=buckets,
        filters=filters,
        extra_where=extra_where,
    )
    grouping_results = sess.exec(grouping_query).fetchall()
    return QueryDistribution(
        results=[
            QueryDistributionGroup(
                group=grouping,
                count=count,
            )
            for grouping, count in grouping_results
        ],
    )


class QueryScatterPoint(BaseModel):
    x: float
    y: float
    n: int


class QueryScatter(BaseModel):
    samples: List[QueryScatterPoint]


def select_for_query_attr_scatter(
    dialect: Dialect,
    tables: Tables,
    project_filters: ProjectFilters,
    x_metric_name: str,
    y_metric_name: str,
    buckets: Literal[10, 100, 1000],
    filters: Optional[search_query.SearchFilters],
    extra_where: Optional[list],
) -> "Select[Tuple[float, float, int]]":
    domain_tables = tables.primary
    x_attr = get_metric_or_enum(
        dialect, domain_tables.analytics, x_metric_name, domain_tables.metrics, {}, buckets=buckets  # type: ignore
    )
    y_attr = get_metric_or_enum(
        dialect, domain_tables.analytics, y_metric_name, domain_tables.metrics, {}, buckets=buckets  # type: ignore
    )
    where = search_query.search_filters(
        tables=tables,
        base=domain_tables.analytics,
        search=filters,
        project_filters=project_filters,
    )
    return (
        select(
            x_attr.group_attr.label("x"),  # type: ignore
            y_attr.group_attr.label("y"),  # type: ignore
            sql_count().label("n"),  # type: ignore
        )
        .where(*where, is_not(x_attr.filter_attr, None), is_not(x_attr.filter_attr, None), *(extra_where or []))
        .group_by("x", "y")
    )


def query_attr_scatter(
    sess: Session,
    tables: Tables,
    project_filters: ProjectFilters,
    x_metric_name: str,
    y_metric_name: str,
    buckets: Literal[10, 100, 1000],
    filters: Optional[search_query.SearchFilters],
    extra_where: Optional[list] = None,
) -> QueryScatter:
    scatter_query = select_for_query_attr_scatter(
        dialect=sess.bind.dialect,  # type: ignore
        tables=tables,
        project_filters=project_filters,
        x_metric_name=x_metric_name,
        y_metric_name=y_metric_name,
        buckets=buckets,
        filters=filters,
        extra_where=extra_where,
    )
    scatter_results = sess.exec(scatter_query).fetchall()

    return QueryScatter(
        samples=[QueryScatterPoint(x=x, y=y, n=n) for x, y, n in scatter_results],
    )


class Query2DEmbedding(BaseModel):
    count: int
    reductions: List[QueryScatterPoint]


TExtraSelect = TypeVar("TExtraSelect", bound=tuple)


def select_for_query_reduction_scatter(
    dialect: Dialect,
    tables: Tables,
    project_filters: ProjectFilters,
    buckets: Literal[10, 100, 1000],
    filters: Optional[search_query.SearchFilters],
    extra_where: Optional[list],
    extra_select: TExtraSelect,
) -> "Select[Union[Tuple[float, float, int], Tuple[float, float, int, TExtraSelect]]]":
    domain_tables = tables.primary
    x_attr = get_float_attr_bucket(dialect, domain_tables.reduction.x, buckets)
    y_attr = get_float_attr_bucket(dialect, domain_tables.reduction.y, buckets)
    where = search_query.search_filters(
        tables=tables,
        base=domain_tables.reduction,
        search=filters,
        project_filters=project_filters,
    )
    return (
        select(
            x_attr.label("xv"),  # type: ignore
            y_attr.label("yv"),  # type: ignore
            sql_count().label("n"),  # type: ignore
            *extra_select,
        )
        .where(
            *where,
            *(extra_where or []),
        )
        .group_by("xv", "yv")
    )


def query_reduction_scatter(
    sess: Session,
    tables: Tables,
    project_filters: ProjectFilters,
    buckets: Literal[10, 100, 1000],
    filters: Optional[search_query.SearchFilters],
    extra_where: Optional[list] = None,
) -> Query2DEmbedding:
    query: "Select[Tuple[float, float, int]]" = select_for_query_reduction_scatter(  # type: ignore
        dialect=sess.bind.dialect,  # type: ignore
        tables=tables,
        project_filters=project_filters,
        buckets=buckets,
        filters=filters,
        extra_where=extra_where,
        extra_select=tuple(),
    )
    results = sess.exec(query).fetchall()
    return Query2DEmbedding(
        count=sum(n for x, y, n in results),
        reductions=[
            QueryScatterPoint(x=x if not math.isnan(x) else 0, y=y if not math.isnan(y) else 0, n=n)
            for x, y, n in results
        ],
    )
