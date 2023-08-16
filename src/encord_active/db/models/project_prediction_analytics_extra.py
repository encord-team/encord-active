from typing import Dict, Optional
from uuid import UUID

from sqlmodel import CheckConstraint, SQLModel

from encord_active.db.models.project_prediction import ProjectPrediction
from encord_active.db.models.project_prediction_analytics import (
    ProjectPredictionAnalytics,
)
from encord_active.db.util.fields import (
    EmbeddingVector,
    field_char8,
    field_embedding_vector,
    field_int,
    field_string_dict,
    field_uuid,
    fk_constraint,
)


class ProjectPredictionAnalyticsExtra(SQLModel, table=True):
    __tablename__ = "prediction_analytics_extra"
    prediction_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    project_hash: UUID = field_uuid()

    # Embeddings
    embedding_clip: Optional[EmbeddingVector] = field_embedding_vector(512, nullable=True)
    embedding_hu: Optional[EmbeddingVector] = field_embedding_vector(7, nullable=True)
    # Metric comments
    metric_metadata: Optional[Dict[str, str]] = field_string_dict(nullable=True)

    __table_args__ = (
        fk_constraint(
            ["prediction_hash", "du_hash", "frame", "annotation_hash"],
            ProjectPredictionAnalytics,
            "fk_prediction_analytics_extra",
        ),
        fk_constraint(["prediction_hash", "project_hash"], ProjectPrediction, "fk_prediction_analytics_extra_project"),
        CheckConstraint("frame >= 0", name="prediction_analytics_frame"),
    )
