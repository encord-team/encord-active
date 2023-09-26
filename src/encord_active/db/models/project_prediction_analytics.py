from typing import Optional
from uuid import UUID

from sqlmodel import CheckConstraint, Index, SQLModel

from encord_active.db.enums import AnnotationType
from encord_active.db.metrics import (
    CUSTOM_METRIC_COUNT,
    AnnotationMetrics,
    assert_cls_metrics_match,
    define_metric_indices,
)
from encord_active.db.models import ProjectCollaborator
from encord_active.db.models.project_prediction import ProjectPrediction
from encord_active.db.util.fields import (
    check_annotation_type,
    field_annotation_type,
    field_bool,
    field_char8,
    field_int,
    field_metric_normal,
    field_metric_positive_float,
    field_metric_positive_integer,
    field_uuid,
    fk_constraint,
)


@assert_cls_metrics_match(AnnotationMetrics, CUSTOM_METRIC_COUNT)
class ProjectPredictionAnalytics(SQLModel, table=True):
    __tablename__ = "prediction_analytics"
    prediction_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    feature_hash: str = field_char8()
    project_hash: UUID = field_uuid()

    # Prediction data for display.
    annotation_type: AnnotationType = field_annotation_type()
    annotation_invalid: bool = field_bool()
    annotation_manual: bool = field_bool()
    annotation_user_id: Optional[int] = field_int(nullable=True)

    # Prediction metadata: (for iou dependent TP vs FP split & feature hash grouping)
    match_annotation_hash: Optional[str] = field_char8(nullable=True)
    match_feature_hash: Optional[str] = field_char8(nullable=True)
    match_duplicate_iou: float = field_metric_normal(nullable=False)  # -1 is always TP
    iou: float = field_metric_normal(nullable=False)

    # Prediction object metrics
    # Metrics - Absolute Size
    metric_width: Optional[int] = field_metric_positive_integer()
    metric_height: Optional[int] = field_metric_positive_integer()
    metric_area: Optional[int] = field_metric_positive_integer()
    # Metrics - Relative Size
    metric_area_relative: Optional[float] = field_metric_normal()
    metric_aspect_ratio: Optional[float] = field_metric_positive_float()
    # Metrics Color
    metric_brightness: Optional[float] = field_metric_normal()
    metric_contrast: Optional[float] = field_metric_normal()
    metric_sharpness: Optional[float] = field_metric_normal()
    metric_red: Optional[float] = field_metric_normal()
    metric_green: Optional[float] = field_metric_normal()
    metric_blue: Optional[float] = field_metric_normal()
    # Random
    metric_random: Optional[float] = field_metric_normal()
    # Both - Annotation based
    metric_annotation_quality: Optional[float] = field_metric_normal()
    # Metrics - Label Only
    metric_max_iou: Optional[float] = field_metric_normal()
    metric_border_relative: Optional[float] = field_metric_normal()
    metric_polygon_similarity: Optional[float] = field_metric_normal()
    metric_missing_or_broken_track: Optional[float] = field_metric_normal()
    metric_inconsistent_class: Optional[float] = field_metric_normal()
    metric_shape_outlier: Optional[float] = field_metric_normal()
    metric_confidence: float = field_metric_normal()

    # 4x custom normal metrics
    metric_custom0: Optional[float] = field_metric_normal()
    metric_custom1: Optional[float] = field_metric_normal()
    metric_custom2: Optional[float] = field_metric_normal()
    metric_custom3: Optional[float] = field_metric_normal()

    __table_args__ = define_metric_indices(
        "prediction_analytics",
        AnnotationMetrics,
        [
            fk_constraint(["prediction_hash", "project_hash"], ProjectPrediction, "fk_prediction_analytics"),
            fk_constraint(
                ["project_hash", "annotation_user_id"],
                ProjectCollaborator,
                "fk_prediction_analytics_annotation_user_id",
                ["project_hash", "user_id"],
            ),
            Index(
                "ix_prediction_analytics_ph_fh_mtc_confidence",
                "prediction_hash",
                "feature_hash",
                "metric_confidence",
            ),
            CheckConstraint("iou BETWEEN 0.0 AND 1.0", name="prediction_analytics_iou"),
            CheckConstraint(
                "match_duplicate_iou BETWEEN 0.0 AND 1.0 OR match_duplicate_iou = -1.0",
                name="prediction_analytics_duplicate_iou",
            ),
            CheckConstraint("frame >= 0", name="prediction_analytics_frame"),
            check_annotation_type("prediction_analytics_annotation_type"),
        ],
        grouping="prediction_hash",
    )
