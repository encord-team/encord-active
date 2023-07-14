from typing import Union, Literal

from sqlalchemy.sql.operators import between_op, in_op

from encord_active.server.routers.queries.domain_query import Tables, SearchFilters, DomainTables, DomainSearchFilters, \
    AnalyticsTable, ReductionTable


def search_filters(
    tables: Tables,
    base: Literal["analytics", "reduction"],
    search: SearchFilters,
) -> list:
    filters = []
    if tables.annotation is None:
        # Data domain only
        if search.annotation is not None:
            raise ValueError(f"Annotation queries are not supported in the raw data domain")
        base_table = tables.data.analytics if base == "analytics" else tables.data.reduction
        _append_filters(
            tables=tables.data,
            search=search.data,
            base_table=base_table,
            filters=filters,
        )
    else:
        # We assume that join
        base_table = tables.annotation.analytics if base == "analytics" else tables.annotation.reduction
        _append_filters(
            tables=tables.annotation,
            search=search.annotation,
            base_table=base_table,
            filters=filters,
        )
        _append_filters(
            tables=tables.data,
            search=search.data,
            base_table=base_table,
            filters=filters,
        )

    return filters


def _append_filters(
    tables: DomainTables,
    search: DomainSearchFilters,
    base_table: Union[AnalyticsTable, ReductionTable],
    filters: list
) -> None:

    # Metric filters
    analytics_join = False
    if len(search.enums) > 0:
        if base_table != tables.analytics:
            for j in tables.join:
                filters.append(
                    getattr(base_table, j) == getattr(tables.analytics, j)
                )
            analytics_join = True
        for metric_name, (filter_start, filter_end) in search.metrics.items():
            if metric_name not in tables.metrics:
                raise ValueError(f"Invalid metric filter: {metric_name}")
            metric_attr = getattr(tables.analytics, metric_name)
            filters.append(between_op(metric_attr, filter_start, filter_end))

    # Enum filters
    if len(search.enums) > 0:
        if base_table != tables.analytics and not analytics_join:
            for j in tables.join:
                filters.append(
                    getattr(base_table, j) == getattr(tables.analytics, j)
                )
        for enum_name, enum_list in search.enums.items():
            if enum_name not in tables.enums:
                raise ValueError(f"Invalid enum filter: {enum_name}")
            enum_attr = getattr(tables.analytics, enum_name)
            filters.append(in_op(enum_attr, enum_list))

    # Tag filters
    if search.tags is not None:
        for j in tables.join:
            filters.append(
                getattr(base_table, j) == getattr(tables.tag, j)
            )
        filters.append(in_op(tables.tag.tag_hash, search.tags))

    # Embedding filters
    if search.reduction is not None:
        if base_table != tables.reduction:
            for j in tables.join:
                filters.append(
                    getattr(base_table, j) == getattr(tables.reduction, j)
                )
        min_x, min_y = search.reduction.min
        max_x, max_y = search.reduction.max
        filters.append(between_op(tables.reduction.x, min_x, max_x))
        filters.append(between_op(tables.reduction.y, min_y, max_y))
