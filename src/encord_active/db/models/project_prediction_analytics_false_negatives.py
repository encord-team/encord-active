from uuid import UUID

from sqlmodel import CheckConstraint, Index, SQLModel

from encord_active.db.models.project_annotation_analytics import (
    ProjectAnnotationAnalytics,
)
from encord_active.db.models.project_prediction import ProjectPrediction
from encord_active.db.util.fields import (
    field_char8,
    field_int,
    field_metric_normal,
    field_uuid,
    fk_constraint,
)


class ProjectPredictionAnalyticsFalseNegatives(SQLModel, table=True):
    __tablename__ = "prediction_analytics_fn"
    prediction_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    project_hash: UUID = field_uuid()

    # IOU threshold for missed prediction
    # this entry is a false negative IFF (iou < iou_threshold)
    # -1.0 is used for unconditional
    iou_threshold: float = field_metric_normal(nullable=False)

    # Associated feature hash - used for common queries so split out explicitly.
    feature_hash: str = field_char8(primary_key=True)

    __table_args__ = (
        fk_constraint(
            ["project_hash", "du_hash", "frame", "annotation_hash"],
            ProjectAnnotationAnalytics,
            "fk_prediction_analytics_fn",
        ),
        fk_constraint(["prediction_hash", "project_hash"], ProjectPrediction, "fk_prediction_analytics_fn_project"),
        Index("fk_prediction_analytics_fn_feature_hash", "prediction_hash", "feature_hash"),
        CheckConstraint(
            "iou_threshold BETWEEN 0.0 AND 1.0 OR iou_threshold = -1.0",
            name="prediction_analytics_fn_iou_threshold",
        ),
        CheckConstraint("frame >= 0", name="prediction_analytics_fn_frame"),
    )
