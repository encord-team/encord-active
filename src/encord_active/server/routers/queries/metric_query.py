import dataclasses
import math
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from fastapi import Depends, Query
from pydantic import BaseModel
from sqlalchemy import Numeric
from sqlalchemy.engine import Engine
from sqlalchemy.sql.functions import count as sql_count_raw
from sqlalchemy.sql.functions import max as sql_max_raw
from sqlalchemy.sql.functions import min as sql_min_raw
from sqlalchemy.sql.functions import sum as sql_sum_raw
from sqlalchemy.sql.operators import between_op, is_not, not_between_op
from sqlmodel import Session, SQLModel, func, select

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


def literal_bucket_depends(default: int) -> Literal[10, 100, 1000]:
    def _parse_buckets(
        buckets: Literal["10", "100", "1000"] = Query(default, alias="buckets")
    ) -> Literal[10, 100, 1000]:
        return int(buckets)  # type: ignore

    return Depends(_parse_buckets, use_cache=False)


def get_metric_or_enum(
    engine: Engine,
    table: AnalyticsTable,
    attr_name: str,
    metrics: Dict[str, MetricDefinition],
    enums: Dict[str, EnumDefinition],
    buckets: Optional[Literal[10, 100, 1000]] = None,
) -> AttrMetadata:
    metric = metrics.get(attr_name, None)
    round_digits = None if buckets is None else int(math.log10(buckets))
    if metric is not None:
        raw_metric_attr = getattr(table, attr_name)
        if metric.type == MetricType.NORMAL:
            if engine.dialect == "sqlite":
                group_attr = raw_metric_attr if round_digits is None else func.ROUND(raw_metric_attr, round_digits)
            else:
                group_attr = (
                    raw_metric_attr
                    if round_digits is None
                    else raw_metric_attr.cast(Numeric(1 + round_digits, round_digits))
                )
            return AttrMetadata(
                group_attr=group_attr,  # type: ignore
                filter_attr=raw_metric_attr,
                metric_type=metric.type,
            )
        elif metric.type == MetricType.UFLOAT:
            # FIXME: work for different distributions (currently ONLY aspect ratio)
            #  hence we can assume value is near 1.0 (so we currently use same rounding as normal.
            group_attr = (
                raw_metric_attr
                if round_digits is None
                else func.ROUND(raw_metric_attr.cast(Numeric(20, 10)), round_digits)
            )
            return AttrMetadata(
                group_attr=group_attr,  # type: ignore
                filter_attr=raw_metric_attr,
                metric_type=metric.type,
            )
        elif metric.type == MetricType.UINT:
            # FIXME: width / height / area / object_count
            #  we currently assume that this will naturally group into sane values
            #  so no post-processing is done here.
            #  may be worth considering rounding to nearest 10 or 100 or selecting 0 -> max to dynamically select?
            return AttrMetadata(
                group_attr=raw_metric_attr,
                filter_attr=raw_metric_attr,
                metric_type=metric.type,
            )
        elif metric.type == MetricType.RANK:
            # FIXME: currently image difficulty - try and see if this metric type can be removed.
            # rank_attr = func.row_number().over(
            #     order_by=raw_metric_attr,
            # )
            return AttrMetadata(
                group_attr=raw_metric_attr,
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
    if metric.type == MetricType.RANK:
        return QueryMetricSummary(
            min=0,
            q1=metric_count // 4,
            median=metric_count // 2,
            q3=(metric_count * 3) // 4,
            max=metric_count,
            count=metric_count,
            moderate=0,
            severe=0,
        )
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


class QuerySummary(BaseModel):
    count: int
    metrics: Dict[str, QueryMetricSummary]
    enums: Dict[str, None]


def query_attr_summary(
    sess: Session,
    tables: Tables,
    project_filters: ProjectFilters,
    filters: Optional[search_query.SearchFilters],
) -> QuerySummary:
    domain_tables = tables.annotation or tables.data
    where = search_query.search_filters(
        tables=tables,
        base="analytics",
        search=filters,
        project_filters=project_filters,
    )
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
        enums={k: None for k, e in domain_tables.enums.items()},  # FIXME: implement properly
    )


class QueryDistributionGroup(BaseModel):
    group: Union[float, str, bool]
    count: int


class QueryDistribution(BaseModel):
    results: List[QueryDistributionGroup]


def query_attr_distribution(
    sess: Session,
    tables: Tables,
    project_filters: ProjectFilters,
    attr_name: str,
    buckets: Literal[10, 100, 1000],
    filters: Optional[search_query.SearchFilters],
) -> QueryDistribution:
    domain_tables = tables.annotation or tables.data
    attr = get_metric_or_enum(
        sess.bind,  # type: ignore
        domain_tables.analytics,
        attr_name,
        domain_tables.metrics,
        domain_tables.enums,
        buckets=buckets,
    )
    where = search_query.search_filters(
        tables=tables,
        base="analytics",
        search=filters,
        project_filters=project_filters,
    )
    grouping_query = (
        select(
            attr.group_attr.label("g"),  # type: ignore
            sql_count(),
        )
        .where(*where, is_not(attr.filter_attr, None))
        .group_by("g")
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


def query_attr_scatter(
    sess: Session,
    tables: Tables,
    project_filters: ProjectFilters,
    x_metric_name: str,
    y_metric_name: str,
    buckets: Literal[10, 100, 1000],
    filters: Optional[search_query.SearchFilters],
) -> QueryScatter:
    domain_tables = tables.annotation or tables.data
    x_attr = get_metric_or_enum(
        sess.bind, domain_tables.analytics, x_metric_name, domain_tables.metrics, {}, buckets=buckets  # type: ignore
    )
    y_attr = get_metric_or_enum(
        sess.bind, domain_tables.analytics, y_metric_name, domain_tables.metrics, {}, buckets=buckets  # type: ignore
    )
    where = search_query.search_filters(
        tables=tables,
        base="analytics",
        search=filters,
        project_filters=project_filters,
    )
    scatter_query = (
        select(
            x_attr.group_attr.label("x"),  # type: ignore
            y_attr.group_attr.label("y"),  # type: ignore
            sql_count(),
        )
        .where(
            *where,
            is_not(x_attr.filter_attr, None),
            is_not(x_attr.filter_attr, None),
        )
        .group_by("x", "y")
    )
    scatter_results = sess.exec(scatter_query).fetchall()

    return QueryScatter(
        samples=[QueryScatterPoint(x=x, y=y, n=n) for x, y, n in scatter_results],
    )


TSearch = TypeVar("TSearch", bound=Tuple)
