from typing import Optional
from uuid import UUID

from sqlmodel import CheckConstraint, SQLModel

from encord_active.analysis.types import EmbeddingDistanceMetric
from encord_active.db.models.project_prediction import ProjectPrediction
from encord_active.db.models.project_prediction_analytics import (
    ProjectPredictionAnalytics,
)
from encord_active.db.util.fields import (
    field_char8,
    field_embedding_distance_metric,
    field_int,
    field_real,
    field_small_int,
    field_uuid,
    fk_constraint,
)


class ProjectPredictionAnalyticsDerived(SQLModel, table=True):
    __tablename__ = "prediction_analytics_derived"
    # Base primary key
    prediction_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    distance_metric: EmbeddingDistanceMetric = field_embedding_distance_metric(primary_key=True)
    distance_index: int = field_small_int(primary_key=True)
    # Extra state
    project_hash: UUID = field_uuid()
    # Query result
    similarity: Optional[float] = field_real()
    dep_du_hash: UUID = field_uuid()
    dep_frame: int = field_int()
    dep_annotation_hash: str = field_char8()

    __table_args__ = (
        fk_constraint(
            ["prediction_hash", "du_hash", "frame", "annotation_hash"],
            ProjectPredictionAnalytics,
            "fk_prediction_analytics_derived",
        ),
        fk_constraint(["prediction_hash", "project_hash"], ProjectPrediction, "fk_prediction_analytics_derive_project"),
        CheckConstraint("frame >= 0", name="prediction_derived_frame"),
        CheckConstraint("similarity >= 0", name="prediction_derived_similarity"),
    )
