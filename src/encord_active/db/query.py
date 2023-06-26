from typing import Union

from sqlalchemy import func
from sqlmodel import Session, select
from .models import ProjectDataAnalytics, ProjectObjectAnalytics


def analysis_count(session: Session, label: bool, project_hash: str) -> int:
    ty = ProjectObjectAnalytics if label else ProjectDataAnalytics
    count: int = session.query(ty).with_entities(func.count()).where(ty.project_hash == project_hash).scalar()
    return count


def analysis_quartile(
    session: Session,
    label: bool,
    project_hash: str,
    metric_name: str,
    quartile: int
) -> Union[float, int]:
    if not metric_name.startswith("metric_"):
        raise ValueError(f"Invalid metric name: {metric_name}")
    ty = ProjectObjectAnalytics if label else ProjectDataAnalytics
    metric_attr = getattr(ty, metric_name)
    count: int = session.query(ty)\
        .with_entities(func.count())\
        .where(ty.project_hash == project_hash)\
        .where(metric_attr is not None).scalar()
    if quartile == 2:
        offset: int = count // 2
    else:
        offset: int = (count * quartile) // 4
    return session.query(ty).with_entities(metric_attr)\
        .where(metric_attr is not None).offset(offset).limit(1).scalar()

