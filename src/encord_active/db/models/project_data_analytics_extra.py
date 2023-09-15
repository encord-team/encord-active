from typing import Dict, Optional
from uuid import UUID

from sqlmodel import CheckConstraint, SQLModel

from encord_active.db.models.project_data_analytics import ProjectDataAnalytics
from encord_active.db.util.fields import (
    EmbeddingVector,
    field_embedding_vector,
    field_int,
    field_string_dict,
    field_uuid,
    fk_constraint,
)


class ProjectDataAnalyticsExtra(SQLModel, table=True):
    __tablename__ = "project_analytics_data_extra"
    # Base primary key
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    # Embeddings
    embedding_clip: Optional[EmbeddingVector] = field_embedding_vector(512, nullable=True)
    # Metric comments
    metric_metadata: Optional[Dict[str, str]] = field_string_dict(nullable=True)

    __table_args__ = (
        fk_constraint(["project_hash", "du_hash", "frame"], ProjectDataAnalytics, "fk_project_analytics_data_extra"),
        CheckConstraint("frame >= 0", name="project_analytics_data_extra_frame"),
    )
