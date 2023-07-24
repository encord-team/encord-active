import uuid
from typing import Dict, List, Literal, Optional, Tuple, Union

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel, Json, ValidationError, parse_obj_as
from sqlalchemy.sql.operators import between_op, in_op

from encord_active.server.routers.queries.domain_query import (
    AnalyticsTable,
    DomainTables,
    ProjectFilters,
    ReductionTable,
    Tables,
)


class Embedding2DFilter(BaseModel):
    reduction_hash: uuid.UUID
    min: Tuple[float, float]
    max: Tuple[float, float]


class DomainSearchFilters(BaseModel):
    metrics: Dict[str, Tuple[float, float]]
    enums: Dict[str, List[str]]
    reduction: Optional[Embedding2DFilter]
    tags: Optional[List[str]]


class SearchFilters(BaseModel):
    data: Optional[DomainSearchFilters]
    annotation: Optional[DomainSearchFilters]


SearchFiltersFastAPI = Optional[SearchFilters]


def parse_search_filters_fast_api(
    value: Optional[Json] = Query(None, alias="filters", description="Search Filters")
) -> SearchFiltersFastAPI:
    try:
        if value is None:
            return None
        else:
            return parse_obj_as(SearchFilters, value)
    except ValidationError as err:
        raise HTTPException(400, detail=err.errors())


SearchFiltersFastAPIDepends = Depends(parse_search_filters_fast_api, use_cache=False)


def search_filters(
    tables: Tables,
    base: Literal["analytics", "reduction"],
    search: Optional[SearchFilters],
    project_filters: ProjectFilters,
) -> list:
    filters: list = []
    if tables.annotation is None:
        base_table = tables.data.analytics if base == "analytics" else tables.data.reduction
        _project_filters(table=base_table, project_filters=project_filters, filters=filters)
        # Data only.
        if search is not None:
            if search.annotation is not None:
                raise ValueError("Annotation queries are not supported in the raw data domain")
            if search.data is not None:
                _append_filters(
                    tables=tables.data,
                    search=search.data,
                    base_table=base_table,
                    filters=filters,
                )
    else:
        base_table = tables.annotation.analytics if base == "analytics" else tables.annotation.reduction
        _project_filters(table=base_table, project_filters=project_filters, filters=filters)
        # Data & Annotation
        if search is not None:
            if search.annotation is not None:
                _append_filters(
                    tables=tables.annotation,
                    search=search.annotation,
                    base_table=base_table,
                    filters=filters,
                )
            if search.data is not None:
                _append_filters(
                    tables=tables.data,
                    search=search.data,
                    base_table=base_table,
                    filters=filters,
                )

    return filters


def _project_filters(
    table: Union[AnalyticsTable, ReductionTable],
    project_filters: ProjectFilters,
    filters: list,
) -> None:
    for k, v in project_filters.items():
        opt_attr = getattr(table, k, None)
        if opt_attr is not None:
            if len(v) == 1:
                filters.append(opt_attr == v[0])
            else:
                filters.append(in_op(opt_attr, v))


def _append_filters(
    tables: DomainTables, search: DomainSearchFilters, base_table: Union[AnalyticsTable, ReductionTable], filters: list
) -> None:
    # Metric filters
    analytics_join = False
    if len(search.metrics) > 0:
        if base_table != tables.analytics:
            for j in tables.join:
                filters.append(getattr(base_table, j) == getattr(tables.analytics, j))
            analytics_join = True
        for metric_name, (filter_start, filter_end) in search.metrics.items():
            if metric_name not in tables.metrics:
                raise ValueError(f"Invalid metric filter: {metric_name}")
            metric_attr = getattr(tables.analytics, metric_name)
            if filter_start == filter_end:
                filters.append(metric_attr == filter_start)
            else:
                filters.append(between_op(metric_attr, filter_start, filter_end))

    # Enum filters
    if len(search.enums) > 0:
        if base_table != tables.analytics and not analytics_join:
            for j in tables.join:
                filters.append(getattr(base_table, j) == getattr(tables.analytics, j))
        for enum_name, enum_list in search.enums.items():
            if enum_name not in tables.enums:
                raise ValueError(f"Invalid enum filter: {enum_name}")
            enum_attr = getattr(tables.analytics, enum_name)
            if len(enum_list) == 1:
                filters.append(enum_attr == enum_list[0])
            else:
                filters.append(in_op(enum_attr, enum_list))

    # Tag filters
    if search.tags is not None:
        for j in tables.join:
            filters.append(getattr(base_table, j) == getattr(tables.tag, j))
        if len(search.tags) == 1:
            filters.append(tables.tag.tag_hash == search.tags[0])
        else:
            filters.append(in_op(tables.tag.tag_hash, search.tags))

    # Embedding filters
    if search.reduction is not None:
        if base_table != tables.reduction:
            for j in tables.join:
                filters.append(getattr(base_table, j) == getattr(tables.reduction, j))
        min_x, min_y = search.reduction.min
        max_x, max_y = search.reduction.max
        filters.append(between_op(tables.reduction.x, min_x, max_x))
        filters.append(between_op(tables.reduction.y, min_y, max_y))
