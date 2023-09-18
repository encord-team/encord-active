from typing import Optional
from uuid import UUID

from sqlmodel import CheckConstraint, SQLModel

from encord_active.analysis.types import EmbeddingDistanceMetric
from encord_active.db.models.project_data_analytics import ProjectDataAnalytics
from encord_active.db.util.fields import (
    field_embedding_distance_metric,
    field_int,
    field_real,
    field_small_int,
    field_uuid,
    fk_constraint,
)


class ProjectDataAnalyticsDerived(SQLModel, table=True):
    __tablename__ = "project_analytics_data_derived"
    # Base primary key
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    distance_metric: EmbeddingDistanceMetric = field_embedding_distance_metric(primary_key=True)
    distance_index: int = field_small_int(primary_key=True)
    # Query result
    similarity: Optional[float] = field_real(nullable=True)
    dep_du_hash: UUID = field_uuid()
    dep_frame: int = field_int()

    __table_args__ = (
        fk_constraint(["project_hash", "du_hash", "frame"], ProjectDataAnalytics, "fk_project_analytics_data_derived"),
        CheckConstraint("frame >= 0", name="project_analytics_data_derived_frame"),
    )