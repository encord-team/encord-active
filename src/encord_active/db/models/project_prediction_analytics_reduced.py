from uuid import UUID

from sqlmodel import CheckConstraint, Index, SQLModel

from encord_active.db.models.project_embedding_reduction import (
    ProjectEmbeddingReduction,
)
from encord_active.db.models.project_prediction import ProjectPrediction
from encord_active.db.models.project_prediction_analytics import (
    ProjectPredictionAnalytics,
)
from encord_active.db.util.fields import (
    field_char8,
    field_int,
    field_real,
    field_uuid,
    fk_constraint,
)


class ProjectPredictionAnalyticsReduced(SQLModel, table=True):
    __tablename__ = "prediction_analytics_reduced"
    # Base primary key
    reduction_hash: UUID = field_uuid(primary_key=True)
    prediction_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    project_hash: UUID = field_uuid()
    # 2D embedding
    x: float = field_real()
    y: float = field_real()

    __table_args__ = (
        fk_constraint(
            ["reduction_hash", "project_hash"],
            ProjectEmbeddingReduction,
            "fk_prediction_analytics_reduced",
        ),
        fk_constraint(
            ["prediction_hash", "du_hash", "frame", "annotation_hash"],
            ProjectPredictionAnalytics,
            "fk_prediction_analytics_reduced_data",
        ),
        fk_constraint(
            ["prediction_hash", "project_hash"], ProjectPrediction, "fk_prediction_analytics_reduced_project"
        ),
        Index("ix_prediction_analytics_reduced_x", "reduction_hash", "prediction_hash", "x", "y"),
        Index("ix_prediction_analytics_reduced_y", "reduction_hash", "prediction_hash", "y", "x"),
        CheckConstraint("frame >= 0", name="prediction_analytics_reduced_frame"),
    )
