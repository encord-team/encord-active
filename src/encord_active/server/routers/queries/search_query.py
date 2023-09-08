import uuid
from typing import Dict, List, Optional, Type, Union

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel, Json, ValidationError, parse_obj_as
from sqlalchemy.sql.operators import between_op, in_op
from sqlmodel import select

from encord_active.db.models import ProjectPredictionAnalyticsFalseNegatives
from encord_active.server.routers.queries.domain_query import (
    AnalyticsTable,
    DomainTables,
    ProjectFilters,
    ReductionTable,
    Tables,
    TagTable,
)


class Embedding2DFilter(BaseModel):
    reduction_hash: uuid.UUID
    x1: float
    x2: float
    y1: float
    y2: float


class DomainSearchFilters(BaseModel):
    metrics: Dict[str, List[float]]
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
    base: Union[AnalyticsTable, ReductionTable, TagTable, Type[ProjectPredictionAnalyticsFalseNegatives]],
    search: Optional[SearchFilters],
    project_filters: ProjectFilters,
) -> list:
    filters: dict[
        Union[AnalyticsTable, ReductionTable, TagTable, Type[ProjectPredictionAnalyticsFalseNegatives]], list
    ] = {}
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
        if filter_table != base:
            _project_filters(
                table=filter_table,
                project_filters=project_filters,
                table_filters=filter_list,
            )

    # Compile into sql
    compiled_filters = filters.pop(base, [])
    _project_filters(
        table=base,
        project_filters=project_filters,
        table_filters=compiled_filters,
    )
    base_is_annotation = getattr(base, "annotation_hash", None) is not None
    exists_filters = []
    for sql_table, sql_filters in filters.items():
        sql_table_is_annotation = getattr(sql_table, "annotation_hash", None) is not None
        sql_filters.append(base.du_hash == sql_table.du_hash)
        sql_filters.append(base.frame == sql_table.frame)
        if sql_table_is_annotation and sql_table_is_annotation:
            sql_filters.append(base.annotation_hash == sql_table.annotation_hash)  # type: ignore
        # Append to filters
        if sql_table_is_annotation and not base_is_annotation:
            # Data Result & Annotation Filter -> use Exists
            exists_filters.extend(sql_filters)
        else:
            compiled_filters.extend(sql_filters)

    if len(exists_filters) > 0:
        compiled_filters.append(select(1).where(*exists_filters).exists())

    return compiled_filters


def _project_filters(
    table: Union[AnalyticsTable, ReductionTable, TagTable, Type[ProjectPredictionAnalyticsFalseNegatives]],
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
    filters: dict[
        Union[AnalyticsTable, ReductionTable, TagTable, Type[ProjectPredictionAnalyticsFalseNegatives]], list
    ],
) -> None:
    # Metric filters
    if len(search.metrics) > 0:
        metrics_list = filters.setdefault(tables.analytics, [])
        for metric_name, filter_list in search.metrics.items():
            if metric_name not in tables.metrics:
                raise ValueError(f"Invalid metric filter: {metric_name}")
            metric_attr = getattr(tables.analytics, metric_name)
            for i in range(0, len(filter_list), 2):
                filter_start, *filter_end_opt = filter_list[i : i + 2]
                filter_end = None if len(filter_end_opt) == 0 else filter_end_opt[0]
                if filter_end is None:
                    metrics_list.append(metric_attr >= filter_start)
                elif filter_start == filter_end:
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
        reduction_list.append(tables.reduction.reduction_hash == search.reduction.reduction_hash)
        reduction_list.append(between_op(tables.reduction.x, search.reduction.x1, search.reduction.x2))
        reduction_list.append(between_op(tables.reduction.y, search.reduction.y1, search.reduction.y2))
