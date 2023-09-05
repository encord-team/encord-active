import uuid
from typing import Dict, List, Literal, Optional, Tuple, Union

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel, Json, ValidationError, parse_obj_as
from sqlalchemy.sql.operators import between_op, in_op
from sqlmodel import select

from encord_active.server.routers.queries.domain_query import (
    AnalyticsTable,
    DomainTables,
    ProjectFilters,
    ReductionTable,
    Tables, TagTable,
)


class Embedding2DFilter(BaseModel):
    reduction_hash: uuid.UUID
    min: Tuple[float, float]
    max: Tuple[float, float]


class DomainSearchFilters(BaseModel):
    metrics: Dict[str, Tuple[float, float]]
    enums: Dict[str, List[str]]
    reduction: Optional[Embedding2DFilter]
    tags: Optional[List[uuid.UUID]]


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
    base: Union[AnalyticsTable, ReductionTable, TagTable],
    search: Optional[SearchFilters],
    project_filters: ProjectFilters,
) -> list:
    filters: dict[Union[AnalyticsTable, ReductionTable, TagTable], list] = {}
    if search is not None:
        if search.annotation is not None:
            _append_filters(
                tables=tables.annotation,
                search=search.annotation,
                filters=filters,
            )
        if search.data is not None:
            _append_filters(
                tables=tables.data,
                search=search.data,
                filters=filters,
            )
    for filter_table, filter_list in filters.items():
        _project_filters(
            table=filter_table,
            project_filters=project_filters,
            table_filters=filter_list,
        )

    # Compile into sql
    compiled_filters = filters.pop(base, [])
    base_is_annotation = getattr(base, "object_hash", None) is not None
    exists_filters = []
    for sql_table, sql_filters in filters.items():
        sql_table_is_annotation = getattr(sql_table, "object_hash", None) is not None
        sql_filters.append(base.du_hash == sql_table.du_hash)
        sql_filters.append(base.frame == sql_table.frame)
        if sql_table_is_annotation and sql_table_is_annotation:
            sql_filters.append(base.object_hash == sql_table.object_hash)
        # Append to filters
        if sql_table_is_annotation and not base_is_annotation:
            # Data Result & Annotation Filter -> use Exists
            exists_filters.extend(sql_filters)
        else:
            compiled_filters.extend(sql_filters)

    if len(exists_filters) > 0:
        compiled_filters.append(
            select(1).where(*exists_filters).exists()
        )

    return compiled_filters


def _project_filters(
    table: Union[AnalyticsTable, ReductionTable, TagTable],
    project_filters: ProjectFilters,
    table_filters: list,
) -> None:
    for k, v in project_filters.items():
        opt_attr = getattr(table, k, None)
        if opt_attr is not None:
            if len(v) == 1:
                table_filters.append(opt_attr == v[0])
            else:
                table_filters.append(in_op(opt_attr, v))


def _append_filters(
    tables: DomainTables,
    search: DomainSearchFilters,
    filters: dict[Union[AnalyticsTable, ReductionTable, TagTable], list[bool]]
) -> None:
    # Metric filters
    if len(search.metrics) > 0:
        metrics_list = filters.setdefault(tables.analytics, [])
        for metric_name, (filter_start, filter_end) in search.metrics.items():
            if metric_name not in tables.metrics:
                raise ValueError(f"Invalid metric filter: {metric_name}")
            metric_attr = getattr(tables.analytics, metric_name)
            if filter_start == filter_end:
                metrics_list.append(metric_attr == filter_start)
            else:
                metrics_list.append(between_op(metric_attr, filter_start, filter_end))

    # Enum filters
    if len(search.enums) > 0:
        enums_list = filters.setdefault(tables.analytics, [])
        for enum_name, enum_list in search.enums.items():
            if enum_name not in tables.enums:
                raise ValueError(f"Invalid enum filter: {enum_name}")
            enum_attr = getattr(tables.analytics, enum_name)
            if len(enum_list) == 1:
                enums_list.append(enum_attr == enum_list[0])
            else:
                enums_list.append(in_op(enum_attr, enum_list))

    # Tag filters
    if search.tags is not None:
        tag_list = filters.setdefault(tables.tag, [])
        if len(search.tags) == 1:
            tag_list.append(tables.tag.tag_hash == search.tags[0])
        else:
            tag_list.append(in_op(tables.tag.tag_hash, search.tags))

    # Embedding filters
    if search.reduction is not None:
        reduction_list = filters.setdefault(tables.reduction, [])
        min_x, min_y = search.reduction.min
        max_x, max_y = search.reduction.max
        reduction_list.append(between_op(tables.reduction.x, min_x, max_x))
        reduction_list.append(between_op(tables.reduction.y, min_y, max_y))
