import dataclasses
from typing import Dict, Type, Union, List, Tuple, Optional

from sqlalchemy import func
from sqlalchemy.sql.operators import is_not
from sqlmodel import Session, SQLModel, select

from encord_active.db.metrics import MetricDefinition, MetricType


@dataclasses.dataclass
class AttrMetadata:
    group_attr: Union[int, float]
    filter_attr: Union[int, float]
    metric_type: Optional[MetricType]


def get_metric_or_enum(
    table: Type[SQLModel],
    attr_name: str,
    metrics: Dict[str, MetricDefinition],
    enums: Dict[str, dict],
) -> AttrMetadata:
    metric = metrics.get(attr_name, None)
    if metric is not None:
        raw_metric_attr = getattr(table, attr_name)
        if metric.type == MetricType.NORMAL:
            return AttrMetadata(
                group_attr=func.round(raw_metric_attr, 3),
                filter_attr=raw_metric_attr,
                metric_type=metric.type,
            )
        elif metric.type == MetricType.UFLOAT:
            return AttrMetadata(
                group_attr=func.round(raw_metric_attr, 2),
                filter_attr=raw_metric_attr,
                metric_type=metric.type,
            )
        elif metric.type == MetricType.UINT:
            return AttrMetadata(
                group_attr=raw_metric_attr,
                filter_attr=raw_metric_attr,
                metric_type=metric.type,
            )
        elif metric.type == MetricType.RANK:
            rank_attr = func.row_number().over(
                order_by=raw_metric_attr,
            )
            return AttrMetadata(
                group_attr=rank_attr,
                filter_attr=raw_metric_attr,
                metric_type=metric.type,
            )
        else:
            raise ValueError(f"Unknown metric type")

    enum = enums.get(attr_name, None)
    if enum is not None:
        raw_enum_attr = getattr(table, attr_name)
        return AttrMetadata(
            group_attr=raw_enum_attr,
            filter_attr=raw_enum_attr,
            metric_type=None,
        )
    raise ValueError(f"Attribute: {attr_name} is invalid")


def query_attr_distribution(
    sess: Session,
    table: Type[SQLModel],
    extra: List[Tuple[str, Union[int, float, str]]],
    where: list,
    attr_name: str,
    metrics: Dict[str, MetricDefinition],
    enums: Dict[str, dict],
) -> dict:
    attr = get_metric_or_enum(table, attr_name, metrics, enums)
    grouping_query = select(
        attr.group_attr,
        func.count(),
        *(query, for key, query in extra),
    ).where(
        *where,
        is_not(attr.filter_attr, None)
    ).group_by(attr.group_attr)
    grouping_results = sess.exec(grouping_query).fetchall()
    return {
        "results": [
            {
                "group": grouping,
                "count": count,
                **{k: v for (k,), v in zip(extra, rest)}
            }
            for grouping, count, *rest in grouping_results
        ],
        "sampling": 1.0,
    }


def query_attr_scatter(
    sess: Session,
    table: Type[SQLModel],
    extra: List[Tuple[str, Union[int, float, str]]],
    where: list,
    x_metric_name: str,
    y_metric_name: str,
    metrics: Dict[str, MetricDefinition],
    enums: Dict[str, dict],
) -> dict:
    x_attr = get_metric_or_enum(table, x_metric_name, metrics, enums)
    y_attr = get_metric_or_enum(table, y_metric_name, metrics, enums)
    scatter_query = select(
        x_attr.group_attr,
        y_attr.group_attr,
        func.count(),
        *(query, for key, query in extra),
    ).where(
        *where,
        is_not(x_attr.filter_attr, None),
        is_not(x_attr.filter_attr, None),
    ).group_by(x_attr.group_attr, y_attr.group_attr)
    scatter_results = sess.exec(scatter_query).fetchall()

    return {
        "sampling": 1.0,
        "samples": [
            {
                "x": x,
                "y": y,
                "n": n,
                **{k: v for (k,), v in zip(extra, rest)}
            } for x, y, n, *rest in scatter_results
        ],
    }
