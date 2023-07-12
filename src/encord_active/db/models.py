import enum
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from uuid import UUID

from sqlalchemy import JSON, Column
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.engine import Engine
from sqlmodel import Field, ForeignKeyConstraint, Index, SQLModel, create_engine

from encord_active.db.metrics import (
    AnnotationMetrics,
    DataMetrics,
    MetricDefinition,
    assert_cls_metrics_match,
)


def fk_constraint(
    fields: List[str],
    table: Type[SQLModel],
    name: str,
) -> ForeignKeyConstraint:
    return ForeignKeyConstraint(
        fields,
        [getattr(table, field) for field in fields],
        onupdate="CASCADE",
        ondelete="CASCADE",
        name=name,
    )


class AnnotationType(enum.Enum):
    CLASSIFICATION = "classification"
    BOUNDING_BOX = "bounding_box"
    ROT_BOUNDING_BOX = "rot_bounding_box"
    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    SKELETON = "skeleton"
    BITMASK = "bitmask"


class Project(SQLModel, table=True):
    __tablename__ = "active_project"
    project_hash: UUID = Field(primary_key=True)
    project_name: str
    project_description: str
    project_remote_ssh_key_path: Optional[str] = Field(nullable=True)
    project_ontology: dict = Field(sa_column=Column(JSON))


class ProjectDataMetadata(SQLModel, table=True):
    __tablename__ = "active_project_data"
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    data_hash: UUID = Field(primary_key=True)

    # Metadata
    label_hash: UUID = Field(unique=True)
    dataset_hash: UUID
    num_frames: int = Field(gt=0)
    frames_per_second: Optional[float] = Field()
    dataset_title: str
    data_title: str
    data_type: str
    label_row_json: dict = Field(sa_column=Column(JSON))

    __table_args__ = (fk_constraint(["project_hash"], Project, "active_project_data_project_fk"),)


class ProjectDataUnitMetadata(SQLModel, table=True):
    __tablename__ = "active_project_data_units"
    project_hash: UUID = Field(primary_key=True, foreign_key=Project.project_hash)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    data_hash: UUID
    __table_args__ = (
        Index("active_project_data_units_unique_du_hash_frame", "du_hash", "frame", unique=True),
        fk_constraint(["project_hash", "data_hash"], ProjectDataMetadata, "active_data_unit_data_fk"),
    )
    # Optionally set to support local data (video can be stored as decoded frames or as a video object)
    data_uri: Optional[str] = Field(default=None)
    data_uri_is_video: bool = Field(default=False)
    # Per-frame information about the root cause.
    objects: list = Field(sa_column=Column(JSON))
    classifications: list = Field(sa_column=Column(JSON))


# MetricFieldTypeNormal = Field(ge=0, le=0)
# MetricFieldTypePositiveInteger = Field(ge=0)
# MetricFieldTypePositiveFloat = Field(ge=0)
MetricFieldTypeNormal = Field()
MetricFieldTypePositiveInteger = Field()
MetricFieldTypePositiveFloat = Field()


def define_metric_indices(
    metric_prefix: str,
    metrics: Dict[str, MetricDefinition],
    extra: Iterable[Union[Index, ForeignKeyConstraint]],
) -> Tuple[Union[Index, ForeignKeyConstraint], ...]:
    values: List[Union[Index, ForeignKeyConstraint]] = [
        Index(f"{metric_prefix}_project_hash_{metric_name}_index", "project_hash", metric_name)
        for metric_name, metric_metadata in metrics.items()
    ]
    values = values + list(extra)
    return tuple(values)


@assert_cls_metrics_match(DataMetrics)
class ProjectDataAnalytics(SQLModel, table=True):
    __tablename__ = "active_project_analytics_data"
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    # Metrics - Absolute Size
    metric_width: Optional[int] = MetricFieldTypePositiveInteger
    metric_height: Optional[int] = MetricFieldTypePositiveInteger
    metric_area: Optional[int] = MetricFieldTypePositiveInteger
    # Metrics - Relative Size
    metric_aspect_ratio: Optional[float] = MetricFieldTypePositiveFloat
    # Metrics Color
    metric_brightness: Optional[float] = MetricFieldTypeNormal
    metric_contrast: Optional[float] = MetricFieldTypeNormal
    metric_sharpness: Optional[float] = MetricFieldTypeNormal
    metric_red: Optional[float] = MetricFieldTypeNormal
    metric_green: Optional[float] = MetricFieldTypeNormal
    metric_blue: Optional[float] = MetricFieldTypeNormal
    # Random
    metric_random: Optional[float] = MetricFieldTypeNormal
    # Both - Annotation based
    metric_annotation_quality: Optional[float] = MetricFieldTypeNormal
    # Image Only
    metric_object_count: Optional[int] = MetricFieldTypePositiveInteger
    metric_object_density: Optional[float] = MetricFieldTypeNormal
    metric_image_difficulty: Optional[float]  # FIXME: is the output of this always an integer??
    metric_image_singularity: Optional[float] = MetricFieldTypeNormal

    __table_args__ = define_metric_indices(
        "active_data",
        DataMetrics,
        [fk_constraint(["project_hash", "du_hash", "frame"], ProjectDataUnitMetadata, "active_data_project_data_fk")],
    )


class ProjectDataAnalyticsExtra(SQLModel, table=True):
    __tablename__ = "active_project_analytics_data_extra"
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    # Embeddings
    embedding_clip: Optional[bytes]
    embedding_hu: Optional[bytes]
    # Metric comments
    metric_metadata: dict = Field(sa_column=Column(JSON))

    # FIXME: 2d embedding reduction (should potentially be moved to separate table)
    embedding_clip_2d_x: Optional[float]
    embedding_clip_2d_y: Optional[float]


@assert_cls_metrics_match(AnnotationMetrics)
class ProjectAnnotationAnalytics(SQLModel, table=True):
    __tablename__ = "active_project_analytics_annotation"
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = Field(primary_key=True, min_length=8, max_length=8)
    # Extra properties
    feature_hash: str = Field(min_length=8, max_length=8)
    annotation_type: AnnotationType = Field(sa_column=Column(SQLEnum(AnnotationType)))
    annotation_email: str
    annotation_manual: bool
    # Embeddings
    embedding_clip: Optional[bytes]
    embedding_hu: Optional[bytes]
    # Metrics - Absolute Size
    metric_width: Optional[int] = MetricFieldTypePositiveInteger
    metric_height: Optional[int] = MetricFieldTypePositiveInteger
    metric_area: Optional[int] = MetricFieldTypePositiveInteger
    # Metrics - Relative Size
    metric_area_relative: Optional[float] = MetricFieldTypeNormal
    metric_aspect_ratio: Optional[float] = MetricFieldTypePositiveFloat
    # Metrics Color
    metric_brightness: Optional[float] = MetricFieldTypeNormal
    metric_contrast: Optional[float] = MetricFieldTypeNormal
    metric_sharpness: Optional[float] = MetricFieldTypeNormal
    metric_red: Optional[float] = MetricFieldTypeNormal
    metric_green: Optional[float] = MetricFieldTypeNormal
    metric_blue: Optional[float] = MetricFieldTypeNormal
    # Random
    metric_random: Optional[float] = MetricFieldTypeNormal
    # Both - Annotation based
    metric_annotation_quality: Optional[float] = MetricFieldTypeNormal
    # Metrics - Label Only
    metric_label_duplicates: Optional[float] = MetricFieldTypeNormal
    metric_label_border_closeness: Optional[float] = MetricFieldTypeNormal
    metric_label_poly_similarity: Optional[float] = MetricFieldTypeNormal
    metric_label_missing_or_broken_tracks: Optional[float] = MetricFieldTypeNormal
    metric_label_inconsistent_classification_and_track: Optional[float] = MetricFieldTypeNormal
    metric_label_shape_outlier: Optional[float] = MetricFieldTypeNormal
    metric_label_confidence: float = MetricFieldTypeNormal

    __table_args__ = define_metric_indices(
        "active_label",
        AnnotationMetrics,
        [fk_constraint(["project_hash", "du_hash", "frame"], ProjectDataUnitMetadata, "active_label_project_data_fk")],
    )


class ProjectAnnotationAnalyticsExtra(SQLModel, table=True):
    __tablename__ = "active_project_analytics_annotation_extra"
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = Field(primary_key=True, min_length=8, max_length=8)
    # Embeddings
    embedding_clip: Optional[bytes]
    embedding_hu: Optional[bytes]
    # Metric comments
    metric_metadata: dict = Field(sa_column=Column(JSON))

    # FIXME: 2d embedding reduction (should potentially be moved to separate table)
    embedding_clip_2d_x: Optional[float]
    embedding_clip_2d_y: Optional[float]


class ProjectTag(SQLModel, table=True):
    __tablename__ = "active_project_tags"
    tag_hash: UUID = Field(primary_key=True)
    project_hash: UUID = Field(index=True)
    name: str = Field(min_length=1, max_length=256)
    description: str

    __table_args__ = (fk_constraint(["project_hash"], Project, "active_project_tags_project_fk"),)


class ProjectTaggedDataUnit(SQLModel, table=True):
    __tablename__ = "active_project_tagged_data"
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    tag_hash: UUID = Field(primary_key=True, index=True)

    __table_args__ = (
        fk_constraint(["tag_hash"], ProjectTag, "active_project_tagged_data_units_tag_fk"),
        fk_constraint(
            ["project_hash", "du_hash", "frame"], ProjectDataAnalytics, "active_project_tagged_data_units_analysis_fk"
        ),
    )


class ProjectTaggedAnnotation(SQLModel, table=True):
    __tablename__ = "active_project_tagged_annotation"
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = Field(primary_key=True, min_length=8, max_length=8)
    tag_hash: UUID = Field(primary_key=True, index=True)

    __table_args__ = (
        fk_constraint(["tag_hash"], ProjectTag, "active_project_tagged_objects_tag_fk"),
        fk_constraint(
            ["project_hash", "du_hash", "frame", "object_hash"],
            ProjectAnnotationAnalytics,
            "active_project_tagged_objects_analysis_fk",
        ),
    )


class ProjectPrediction(SQLModel, table=True):
    __tablename__ = "active_project_prediction"
    prediction_hash: UUID = Field(primary_key=True)
    project_hash: UUID
    name: str
    __table_args__ = (fk_constraint(["project_hash"], Project, "active_project_prediction_project_hash_fk"),)


@assert_cls_metrics_match(AnnotationMetrics)
class ProjectPredictionAnalytics(SQLModel, table=True):
    __tablename__ = "active_project_prediction_analytics"
    prediction_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = Field(primary_key=True, min_length=8, max_length=8)
    feature_hash: str = Field(min_length=8, max_length=8)
    project_hash: UUID

    # Prediction data for display.
    annotation_type: AnnotationType = Field(sa_column=Column(SQLEnum(AnnotationType)))

    # Prediction metadata: (for iou dependent TP vs FP split & feature hash grouping)
    match_object_hash: Optional[str] = Field(min_length=8, max_length=8)
    match_feature_hash: Optional[str] = Field(min_length=8, max_length=8)
    match_duplicate_iou: float = Field(ge=-1, le=1)  # 0 -> 1, -1 is special case always match.
    iou: float = Field(ge=0, le=1)

    # Prediction object metrics
    # Metrics - Absolute Size
    metric_width: Optional[int] = MetricFieldTypePositiveInteger
    metric_height: Optional[int] = MetricFieldTypePositiveInteger
    metric_area: Optional[int] = MetricFieldTypePositiveInteger
    # Metrics - Relative Size
    metric_area_relative: Optional[float] = MetricFieldTypeNormal
    metric_aspect_ratio: Optional[float] = MetricFieldTypePositiveFloat
    # Metrics Color
    metric_brightness: Optional[float] = MetricFieldTypeNormal
    metric_contrast: Optional[float] = MetricFieldTypeNormal
    metric_sharpness: Optional[float] = MetricFieldTypeNormal
    metric_red: Optional[float] = MetricFieldTypeNormal
    metric_green: Optional[float] = MetricFieldTypeNormal
    metric_blue: Optional[float] = MetricFieldTypeNormal
    # Random
    metric_random: Optional[float] = MetricFieldTypeNormal
    # Both - Annotation based
    metric_annotation_quality: Optional[float] = MetricFieldTypeNormal
    # Metrics - Label Only
    metric_label_duplicates: Optional[float] = MetricFieldTypeNormal
    metric_label_border_closeness: Optional[float] = MetricFieldTypeNormal
    metric_label_poly_similarity: Optional[float] = MetricFieldTypeNormal
    metric_label_missing_or_broken_tracks: Optional[float] = MetricFieldTypeNormal
    metric_label_inconsistent_classification_and_track: Optional[float] = MetricFieldTypeNormal
    metric_label_shape_outlier: Optional[float] = MetricFieldTypeNormal
    metric_label_confidence: float = MetricFieldTypeNormal

    __table_args__ = (
        fk_constraint(["prediction_hash"], ProjectPrediction, "active_project_prediction_objects_prediction_fk"),
        Index(
            "active_project_prediction_objects_confidence_index",
            "prediction_hash",
            "metric_label_confidence",
        ),
        Index(
            "active_project_prediction_objects_feature_confidence_index",
            "prediction_hash",
            "feature_hash",
            "metric_label_confidence",
        ),
    )


class ProjectPredictionAnnotationExtra(SQLModel, table=True):
    __tablename__ = "active_project_prediction_analytics_extra"
    prediction_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = Field(primary_key=True, min_length=8, max_length=8)

    # Packed prediction bytes, format depends on annotation_type on associated
    # non-extra table.
    # BB = [x1, y1, x2, y2] (np.float)
    # BITMASK = count array (np.int)
    # Polygon = [[x, y], [x,y]] (np.float)
    annotation_bytes: bytes

    # Embeddings
    embedding_clip: Optional[bytes]

    # FIXME: 2d embedding reduction (should potentially be moved to separate table)
    embedding_clip_2d_x: Optional[float]
    embedding_clip_2d_y: Optional[float]


class ProjectPredictionAnalyticsFalseNegatives(SQLModel, table=True):
    __tablename__ = "active_project_prediction_analytics_false_negatives"
    prediction_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = Field(primary_key=True, min_length=8, max_length=8)

    # Unmatched annotation properties
    feature_hash: str = Field(min_length=8, max_length=8)
    annotation_type: AnnotationType = Field(sa_column=Column(SQLEnum(AnnotationType)))
    annotation_email: str
    annotation_manual: bool

    # Unmatched annotation metrics.
    # Metrics - Absolute Size
    metric_width: Optional[int] = MetricFieldTypePositiveInteger
    metric_height: Optional[int] = MetricFieldTypePositiveInteger
    metric_area: Optional[int] = MetricFieldTypePositiveInteger
    # Metrics - Relative Size
    metric_area_relative: Optional[float] = MetricFieldTypeNormal
    metric_aspect_ratio: Optional[float] = MetricFieldTypePositiveFloat
    # Metrics Color
    metric_brightness: Optional[float] = MetricFieldTypeNormal
    metric_contrast: Optional[float] = MetricFieldTypeNormal
    metric_sharpness: Optional[float] = MetricFieldTypeNormal
    metric_red: Optional[float] = MetricFieldTypeNormal
    metric_green: Optional[float] = MetricFieldTypeNormal
    metric_blue: Optional[float] = MetricFieldTypeNormal
    # Random
    metric_random: Optional[float] = MetricFieldTypeNormal
    # Both - Annotation based
    metric_annotation_quality: Optional[float] = MetricFieldTypeNormal
    # Metrics - Label Only
    metric_label_duplicates: Optional[float] = MetricFieldTypeNormal
    metric_label_border_closeness: Optional[float] = MetricFieldTypeNormal
    metric_label_poly_similarity: Optional[float] = MetricFieldTypeNormal
    metric_label_missing_or_broken_tracks: Optional[float] = MetricFieldTypeNormal
    metric_label_inconsistent_classification_and_track: Optional[float] = MetricFieldTypeNormal
    metric_label_shape_outlier: Optional[float] = MetricFieldTypeNormal
    metric_label_confidence: float = MetricFieldTypeNormal

    __table_args__ = (
        fk_constraint(["prediction_hash"], ProjectPrediction, "active_project_prediction_unmatched_prediction_fk"),
        Index("active_project_prediction_unmatched_feature_hash_index", "prediction_hash", "feature_hash"),
    )


# FIXME: metrics should be inline predictions or separate (need same keys for easy comparison)??


_init_metadata: Set[str] = set()


def get_engine(path: Path, concurrent: bool = False) -> Engine:
    override_db = os.environ.get("ENCORD_ACTIVE_DATABASE", None)
    create_db_schema = os.environ.get("ENCORD_ACTIVE_DATABASE_SCHEMA_UPDATE", "1")

    connect_args = {"check_same_thread": False} if concurrent else {}
    path = path.expanduser().resolve()
    print(f"Connection to database: {path}")
    engine = create_engine(override_db if override_db is not None else f"sqlite:///{path}", connect_args=connect_args)
    path_key = path.as_posix()
    if path_key not in _init_metadata and create_db_schema == "1":
        # import encord_active.db.migrations.env as migrate_env
        # FIXME: use alembic to auto-run migrations instead of SQLModel!!
        SQLModel.metadata.create_all(engine)
        _init_metadata.add(path_key)
    return engine
