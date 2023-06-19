from typing import Union

from sqlalchemy import func
from sqlmodel import Session, select
from .models import ProjectDataAnalytics, ProjectLabelAnalytics


def analysis_count(session: Session, label: bool, project_hash: str) -> int:
    ty = ProjectLabelAnalytics if label else ProjectDataAnalytics
    count: int = session.query(ty).with_entities(func.count()).where(ty.project_hash == project_hash).scalar()
    return count


def proj(session: Session, label: bool, project_hash: str, metric_name: str) -> Union[float, int]:
    if not metric_name.startswith("metric_"):
        raise ValueError(f"Invalid metric name: {metric_name}")
    ty = ProjectLabelAnalytics if label else ProjectDataAnalytics
    count: int = session.query(ty)\
        .with_entities(func.count())\
        .where(ty.project_hash == project_hash)\
        .where(getattr(ty, metric_name) is not None).scalar()
    offset: int = count // 2
    session.query(ty).with_entities().offset(offset).limit(1).scalar()
