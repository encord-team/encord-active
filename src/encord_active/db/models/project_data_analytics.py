from typing import Optional
from uuid import UUID

from sqlmodel import CheckConstraint, SQLModel

from encord_active.db.enums import DataType
from encord_active.db.metrics import (
    CUSTOM_METRIC_COUNT,
    DataMetrics,
    assert_cls_metrics_match,
    define_metric_indices,
)
from encord_active.db.models.project_data_unit_metadata import ProjectDataUnitMetadata
from encord_active.db.util.fields import (
    field_data_type,
    field_int,
    field_metric_normal,
    field_metric_positive_float,
    field_metric_positive_integer,
    field_uuid,
    fk_constraint,
)


@assert_cls_metrics_match(DataMetrics, CUSTOM_METRIC_COUNT)
class ProjectDataAnalytics(SQLModel, table=True):
    __tablename__ = "project_analytics_data"
    # Base primary key
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    # Enums:
    data_type: DataType = field_data_type()
    # Metrics - Absolute Size
    metric_width: Optional[int] = field_metric_positive_integer()
    metric_height: Optional[int] = field_metric_positive_integer()
    metric_area: Optional[int] = field_metric_positive_integer()
    # Metrics - Relative Size
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
    # Image Only
    metric_object_count: Optional[int] = field_metric_positive_integer()
    metric_object_density: Optional[float] = field_metric_normal()
    metric_image_difficulty: Optional[float] = field_metric_normal()
    metric_image_uniqueness: Optional[float] = field_metric_normal()

    # 4x custom normal metrics
    metric_custom0: Optional[float] = field_metric_normal()
    metric_custom1: Optional[float] = field_metric_normal()
    metric_custom2: Optional[float] = field_metric_normal()
    metric_custom3: Optional[float] = field_metric_normal()

    __table_args__ = define_metric_indices(
        "project_analytics_data",
        DataMetrics,
        [
            fk_constraint(["project_hash", "du_hash", "frame"], ProjectDataUnitMetadata, "fk_project_analytics_data"),
            CheckConstraint("frame >= 0", name="project_analytics_data_frame"),
        ],
    )
