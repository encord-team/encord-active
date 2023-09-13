import enum
import os
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from uuid import UUID

from psycopg2.extensions import connection as pg_connection
from sqlalchemy import (
    INT,
    JSON,
    REAL,
    SMALLINT,
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Integer,
    LargeBinary,
    event,
)
from sqlalchemy.engine import Engine
from sqlmodel import Field, ForeignKeyConstraint, Index, SQLModel, create_engine
from sqlmodel.sql.sqltypes import GUID, AutoString

from encord_active.analysis.types import EmbeddingDistanceMetric
from encord_active.db.enums import (
    AnnotationType,
    AnnotationTypeMaxValue,
    DataType,
    DataTypeMaxValue,
)
from encord_active.db.metrics import (
    AnnotationMetrics,
    DataMetrics,
    MetricDefinition,
    MetricType,
    assert_cls_metrics_match,
)
from encord_active.db.util.char8 import Char8
from encord_active.db.util.encrypted_str import EncryptedStr
from encord_active.db.util.pgvector import PGVector
from encord_active.db.util.strdict import StrDict


def fk_constraint(
    fields: List[str],
    table: Type[SQLModel],
    name: str,
    foreign_fields: Optional[List[str]] = None,
) -> ForeignKeyConstraint:
    return ForeignKeyConstraint(
        fields,
        [getattr(table, field) for field in foreign_fields or fields],
        onupdate="CASCADE",
        ondelete="CASCADE",
        name=name,
    )


def field_uuid(primary_key: bool = False, unique: bool = False, nullable: bool = False) -> UUID:
    return Field(
        primary_key=primary_key, unique=unique, sa_column=Column(GUID(), nullable=nullable, primary_key=primary_key)
    )


def field_bool(nullable: bool = False) -> bool:
    return Field(sa_column=Column(Boolean, nullable=nullable))


def field_int(primary_key: bool = False, nullable: bool = False) -> int:
    return Field(primary_key=primary_key, sa_column=Column(Integer, nullable=nullable, primary_key=primary_key))


def field_small_int(primary_key: bool = False, nullable: bool = False) -> int:
    return Field(primary_key=primary_key, sa_column=Column(SMALLINT, nullable=nullable, primary_key=primary_key))


def field_big_int(primary_key: bool = False, nullable: bool = False) -> int:
    return Field(primary_key=primary_key, sa_column=Column(BigInteger, nullable=nullable, primary_key=primary_key))


def field_datetime(nullable: bool = False) -> datetime:
    return Field(sa_column=Column(TIMESTAMP, nullable=nullable))


def field_real(nullable: bool = False) -> float:
    return Field(sa_column=Column(REAL, nullable=nullable))


def field_text(nullable: bool = False) -> str:
    return Field(sa_column=Column(AutoString, nullable=nullable))


def field_encrypted_text(nullable: bool = False) -> str:
    return Field(sa_column=Column(EncryptedStr, nullable=nullable))


def field_bytes(nullable: bool = False) -> bytes:
    return Field(sa_column=Column(LargeBinary, nullable=nullable))


def field_json_dict(nullable: bool = False, default: Optional[dict] = None) -> dict:
    return Field(sa_column=Column(JSON, nullable=nullable, default=default))


def field_json_list(nullable: bool = False, default: Optional[list] = None) -> list:
    return Field(sa_column=Column(JSON, nullable=nullable, default=default))


EmbeddingVector = bytes  # FIXME: Union[list[float], bytes]


def field_embedding_vector(dim: int, nullable: bool = False) -> EmbeddingVector:
    return Field(sa_column=Column(PGVector(dim), nullable=nullable))


def field_string_dict(nullable: bool = False) -> dict[str, str]:
    return Field(sa_column=Column(StrDict(), nullable=nullable))


"""
Number of custom metrics supported for each table
"""
CUSTOM_METRIC_COUNT: int = 4


class EmbeddingReductionType(enum.Enum):
    UMAP = "umap"


class ImportMetadataType(enum.Enum):
    COCO = "coco"


class Project(SQLModel, table=True):
    __tablename__ = "project"
    project_hash: UUID = field_uuid(primary_key=True)
    name: str = field_text()
    description: str = field_text()
    ontology: dict = field_json_dict()
    remote: bool = field_bool()

    # Custom metadata, list of all metadata for custom metrics
    custom_metrics: list = field_json_list(default=[])


class ProjectImportMetadata(SQLModel, table=True):
    __tablename__ = "project_import"
    project_hash: UUID = field_uuid(primary_key=True)
    import_metadata: dict = field_json_dict()
    import_metadata_type: ImportMetadataType

    __table_args__ = (fk_constraint(["project_hash"], Project, "fk_project_import"),)


class ProjectEmbeddingIndex(SQLModel, table=True):
    __tablename__ = "project_embedding_index"
    project_hash: UUID = field_uuid(primary_key=True)

    data_index_dirty: bool = field_bool()
    data_index_name: Optional[str] = field_text(nullable=True)
    data_index_compiled: Optional[bytes] = field_bytes(nullable=True)

    annotation_index_dirty: bool = field_bool()
    annotation_index_name: Optional[str] = field_text(nullable=True)
    annotation_index_compiled: Optional[bytes] = field_bytes(nullable=True)

    __table_args__ = (fk_constraint(["project_hash"], Project, "fk_project_embedding_index"),)


class ProjectCollaborator(SQLModel, table=True):
    __tablename__ = "project_collaborator"
    project_hash: UUID = field_uuid(primary_key=True)
    user_id: int = field_big_int(primary_key=True)
    user_email: str = field_text()

    __table_args__ = (fk_constraint(["project_hash"], Project, "fk_project_collaborator"),)


class ProjectEmbeddingReduction(SQLModel, table=True):
    __tablename__ = "project_embedding_reduction"
    reduction_hash: UUID = field_uuid(primary_key=True)
    project_hash: UUID = field_uuid()
    reduction_name: str = field_text()
    reduction_description: str = field_text()

    # Binary encoded information on the 2d reduction implementation.
    reduction_type: EmbeddingReductionType
    reduction_bytes: bytes

    __table_args__ = (
        fk_constraint(["project_hash"], Project, "fk_project_embedding_reduction"),
        Index("ix_project_embedding_reduction", "project_hash", "reduction_hash", unique=True),
    )


class ProjectDataMetadata(SQLModel, table=True):
    __tablename__ = "project_data"
    # Base primary key
    project_hash: UUID = field_uuid(primary_key=True)
    data_hash: UUID = field_uuid(primary_key=True)

    # Metadata
    label_hash: UUID = field_uuid(unique=True)
    dataset_hash: UUID = field_uuid()
    num_frames: int = field_int()
    frames_per_second: Optional[float] = field_real(nullable=True)
    dataset_title: str = field_text()
    data_title: str = field_text()
    data_type: str = field_text()
    created_at: datetime = field_datetime()
    last_edited_at: datetime = field_datetime()
    object_answers: dict = field_json_dict()
    classification_answers: dict = field_json_dict()

    __table_args__ = (
        fk_constraint(["project_hash"], Project, "fk_project_data"),
        CheckConstraint("num_frames > 0", name="project_data_num_frames"),
    )


class ProjectDataUnitMetadata(SQLModel, table=True):
    __tablename__ = "project_data_units"
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    data_hash: UUID = field_uuid()

    # Size metadata - used for efficient computation scheduling.
    width: int = field_int()
    height: int = field_int()

    # Optionally set to support local data (video can be stored as decoded frames or as a video object)
    data_uri: Optional[str] = field_text(nullable=True)
    data_uri_is_video: bool = field_bool()

    # Extra metadata for image groups
    data_title: str = field_text()
    data_type: str = field_text()

    # Per-frame information about the root cause.
    objects: list = field_json_list()
    classifications: list = field_json_list()

    __table_args__ = (
        fk_constraint(["project_hash", "data_hash"], ProjectDataMetadata, "fk_project_data_units"),
        CheckConstraint("frame >= 0", name="project_data_units_frame"),
    )


def field_metric_normal(nullable: bool = True) -> float:
    return Field(
        sa_column=Column(REAL, nullable=nullable),
    )


def field_metric_positive_integer(nullable: bool = True) -> int:
    return Field(sa_column=Column(INT, nullable=nullable))


def field_metric_positive_float(nullable: bool = True) -> float:
    return Field(sa_column=Column(REAL, nullable=nullable))


def field_char8(primary_key: bool = False, nullable: bool = False) -> str:
    return Field(
        primary_key=primary_key,
        sa_column=Column(Char8, nullable=nullable, primary_key=primary_key),
        min_length=8,
        max_length=8,
    )


def field_annotation_type() -> AnnotationType:
    return Field(sa_column=Column(SMALLINT, nullable=False))


def check_annotation_type(name: str) -> CheckConstraint:
    return CheckConstraint(f"annotation_type >= 0 AND annotation_type <= {AnnotationTypeMaxValue}", name=name)


def field_data_type() -> DataType:
    return Field(sa_column=Column(SMALLINT), nullable=False)


def check_data_type(name: str) -> CheckConstraint:
    return CheckConstraint(f"data_type >= 0 AND data_type <= {DataTypeMaxValue}", name=name)


def field_embedding_distance_metric(primary_key: bool = False) -> EmbeddingDistanceMetric:
    return Field(sa_column=Column(SMALLINT, nullable=False, primary_key=primary_key))


def define_metric_indices(
    metric_prefix: str,
    metrics: Dict[str, MetricDefinition],
    extra: Iterable[Union[Index, CheckConstraint, ForeignKeyConstraint]],
    add_constraints: bool = True,
    grouping: str = "project_hash",
) -> Tuple[Union[Index, CheckConstraint, ForeignKeyConstraint], ...]:
    values: List[Union[Index, CheckConstraint, ForeignKeyConstraint]] = [
        Index(f"ix_{metric_prefix}_ph_{metric_name.replace('metric_', 'mtc_')}", grouping, metric_name)
        for metric_name, metric_metadata in metrics.items()
    ]
    if add_constraints:
        for metric_name, metric_metadata in metrics.items():
            name = f"{metric_prefix}_{metric_name.replace('metric_', 'mtc_')}"
            if metric_metadata.type == MetricType.NORMAL:
                values.append(CheckConstraint(f"{metric_name} BETWEEN 0.0 AND 1.0", name=name))
            elif metric_metadata.type == MetricType.UINT:
                values.append(CheckConstraint(f"{metric_name} >= 0", name=name))
            elif metric_metadata.type == MetricType.UFLOAT:
                values.append(CheckConstraint(f"{metric_name} >= 0.0", name=name))
    values = values + list(extra)
    return tuple(values)


@assert_cls_metrics_match(DataMetrics, CUSTOM_METRIC_COUNT)
class ProjectDataAnalytics(SQLModel, table=True):
    __tablename__ = "project_analytics_data"
    # Base primary key
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
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


class ProjectDataAnalyticsReduced(SQLModel, table=True):
    __tablename__ = "project_analytics_data_reduced"
    # Base primary key
    reduction_hash: UUID = field_uuid(primary_key=True)
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    x: float = field_real()
    y: float = field_real()

    __table_args__ = (
        fk_constraint(["project_hash", "du_hash", "frame"], ProjectDataAnalytics, "fk_project_analytics_data_reduced"),
        fk_constraint(["reduction_hash"], ProjectEmbeddingReduction, "fk_project_analytics_data_reduced_reduction"),
        Index("ix_project_analytics_data_reduced_x", "reduction_hash", "project_hash", "x", "y"),
        Index("ix_project_analytics_data_reduced_y", "reduction_hash", "project_hash", "y", "x"),
        CheckConstraint("frame >= 0", name="project_analytics_data_reduced_frame"),
    )


@assert_cls_metrics_match(AnnotationMetrics, CUSTOM_METRIC_COUNT)
class ProjectAnnotationAnalytics(SQLModel, table=True):
    __tablename__ = "project_analytics_annotation"
    # Base primary key
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    # Extra properties
    feature_hash: str = field_char8()
    annotation_type: AnnotationType = field_annotation_type()
    annotation_user_id: Optional[int] = field_int(nullable=True)
    annotation_manual: bool = field_bool()
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
        "project_analytics_annotation",
        AnnotationMetrics,
        [
            fk_constraint(
                ["project_hash", "du_hash", "frame"], ProjectDataUnitMetadata, "fk_project_analytics_annotation"
            ),
            fk_constraint(
                ["project_hash", "annotation_user_id"],
                ProjectCollaborator,
                "fk_project_analytics_annotation_user_id",
                ["project_hash", "user_id"],
            ),
            CheckConstraint("frame >= 0", name="project_analytics_annotation_frame"),
        ],
    )


class ProjectAnnotationAnalyticsExtra(SQLModel, table=True):
    __tablename__ = "project_analytics_annotation_extra"
    # Base primary key
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    # Embeddings
    embedding_clip: Optional[EmbeddingVector] = field_embedding_vector(512, nullable=True)
    embedding_hu: Optional[EmbeddingVector] = field_embedding_vector(7, nullable=True)
    # Metric comments
    metric_metadata: Optional[Dict[str, str]] = field_string_dict(nullable=True)

    __table_args__ = (
        fk_constraint(
            ["project_hash", "du_hash", "frame", "annotation_hash"],
            ProjectAnnotationAnalytics,
            "fk_project_analytics_annotation_extra",
        ),
        CheckConstraint("frame >= 0", name="project_analytics_annotation_extra_frame"),
    )


class ProjectAnnotationAnalyticsDerived(SQLModel, table=True):
    __tablename__ = "project_analytics_annotation_derived"
    # Base primary key
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    distance_metric: EmbeddingDistanceMetric = field_embedding_distance_metric(primary_key=True)
    distance_index: int = field_small_int(primary_key=True)
    # Query result
    similarity: Optional[float] = field_real(nullable=True)
    dep_du_hash: UUID = field_uuid()
    dep_frame: int = field_int()
    dep_annotation_hash: str = field_char8()

    __table_args__ = (
        fk_constraint(
            ["project_hash", "du_hash", "frame", "annotation_hash"],
            ProjectAnnotationAnalytics,
            "fk_project_analytics_annotation_derived",
        ),
        CheckConstraint("frame >= 0", name="project_analytics_annotation_derived_frame"),
    )


class ProjectAnnotationAnalyticsReduced(SQLModel, table=True):
    __tablename__ = "project_analytics_annotation_reduced"
    # Base primary key
    reduction_hash: UUID = field_uuid(primary_key=True)
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    # 2D embedding
    x: float = field_real()
    y: float = field_real()

    __table_args__ = (
        fk_constraint(
            ["project_hash", "du_hash", "frame", "annotation_hash"],
            ProjectAnnotationAnalytics,
            "fk_project_analytics_annotation_reduced",
        ),
        fk_constraint(
            ["reduction_hash"],
            ProjectEmbeddingReduction,
            "fk_project_analytics_annotation_reduced_reduction",
        ),
        Index("ix_project_analytics_annotation_reduced_x", "reduction_hash", "project_hash", "x", "y"),
        Index("ix_project_analytics_annotation_reduced_y", "reduction_hash", "project_hash", "y", "x"),
        CheckConstraint("frame >= 0", name="project_analytics_annotation_reduced_frame"),
    )


class ProjectTag(SQLModel, table=True):
    __tablename__ = "project_tags"
    tag_hash: UUID = field_uuid(primary_key=True)
    project_hash: UUID = field_uuid()
    name: str = field_text()
    description: str = field_text()

    __table_args__ = (fk_constraint(["project_hash"], Project, "fk_project_tags"),)


class ProjectTaggedDataUnit(SQLModel, table=True):
    __tablename__ = "project_tagged_data"
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    tag_hash: UUID = field_uuid(primary_key=True)

    __table_args__ = (
        fk_constraint(["project_hash", "du_hash", "frame"], ProjectDataAnalytics, "fk_project_tagged_data"),
        fk_constraint(["tag_hash"], ProjectTag, "fk_project_tagged_data_tag"),
        CheckConstraint("frame >= 0", name="project_tagged_data_frame"),
    )


class ProjectTaggedAnnotation(SQLModel, table=True):
    __tablename__ = "project_tagged_annotation"
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    tag_hash: UUID = field_uuid(primary_key=True)

    __table_args__ = (
        fk_constraint(
            ["project_hash", "du_hash", "frame", "annotation_hash"],
            ProjectAnnotationAnalytics,
            "fk_project_tagged_annotation",
        ),
        fk_constraint(["tag_hash"], ProjectTag, "fk_project_tagged_annotation_tag"),
        CheckConstraint("frame >= 0", name="project_tagged_annotation_frame"),
    )


class ProjectPrediction(SQLModel, table=True):
    __tablename__ = "prediction"
    prediction_hash: UUID = field_uuid(primary_key=True)
    project_hash: UUID = field_uuid()
    external_project_hash: Optional[UUID] = field_uuid(nullable=True)
    name: str
    __table_args__ = (
        fk_constraint(["project_hash"], Project, "fk_prediction"),
        Index("uq_project_prediction", "project_hash", "prediction_hash", unique=True),
    )


class ProjectPredictionDataMetadata(SQLModel, table=True):
    __tablename__ = "prediction_data"
    # Base primary key
    prediction_hash: UUID = field_uuid(primary_key=True)
    data_hash: UUID = field_uuid(primary_key=True)

    # Associated project_hash
    project_hash: UUID = field_uuid()

    # Metadata
    label_hash: UUID = field_uuid(unique=True)
    created_at: datetime = field_datetime()
    last_edited_at: datetime = field_datetime()
    object_answers: dict = field_json_dict()
    classification_answers: dict = field_json_dict()

    __table_args__ = (
        fk_constraint(["prediction_hash", "project_hash"], ProjectPrediction, "fk_prediction_data"),
        fk_constraint(["project_hash", "data_hash"], ProjectDataMetadata, "fk_prediction_data_project"),
    )


class ProjectPredictionDataUnitMetadata(SQLModel, table=True):
    __tablename__ = "prediction_data_units"
    prediction_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)

    # Associated project_hash / data_hash
    project_hash: UUID = field_uuid()
    data_hash: UUID = field_uuid()

    # Per-frame information about the root cause.
    objects: list = field_json_list()
    classifications: list = field_json_list()

    __table_args__ = (
        fk_constraint(
            ["prediction_hash", "data_hash"],
            ProjectPredictionDataMetadata,
            "fk_prediction_data_units",
        ),
        fk_constraint(["project_hash", "du_hash", "frame"], ProjectDataUnitMetadata, "fk_prediction_data_units_data"),
        fk_constraint(["prediction_hash", "project_hash"], ProjectPrediction, "fk_prediction_data_units_project"),
        CheckConstraint("frame >= 0", name="prediction_data_units_frame"),
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
        ],
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


# FIXME: metrics should be inline predictions or separate (need same keys for easy comparison)??


_init_metadata: Set[str] = set()


@event.listens_for(Engine, "connect")
def set_sqlite_fk_pragma(
    dbapi_connection: Union[pg_connection, SqliteConnection],
    connection_record: None,
) -> None:
    # For sqlite - enable foreign_keys support
    #  (otherwise we need to use complex delete statements)
    if isinstance(dbapi_connection, SqliteConnection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine(
    path: Path,
    concurrent: bool = False,
    use_alembic: bool = True,
) -> Engine:
    override_db = os.environ.get("ENCORD_ACTIVE_DATABASE", None)
    database_echo = os.environ.get("ENCORD_ACTIVE_DATABASE_ECHO", "0") == "1"
    create_db_schema = os.environ.get("ENCORD_ACTIVE_DATABASE_SCHEMA_UPDATE", "1")

    path = path.expanduser().resolve()
    engine_url = override_db if override_db is not None else f"sqlite:///{path}"
    connect_args = {"check_same_thread": False} if concurrent and engine_url.startswith("sqlite:/") else {}

    # Create the engine connection
    print(f"Connection to database: {engine_url}")
    engine = create_engine(engine_url, connect_args=connect_args, echo=database_echo)
    path_key = path.as_posix()
    if path_key not in _init_metadata and create_db_schema == "1":
        if use_alembic:
            # Execute alembic config
            import encord_active.db as alembic_file

            alembic_cwd = Path(alembic_file.__file__).expanduser().resolve().parent
            alembic_args = [
                "alembic",
                "--raiseerr",
                "-x",
                f"dbPath={engine_url}",
                "upgrade",
                "head",
            ]
            # How to run alembic via subprocess if running locally with temp cwd change
            # ends up causing issues:
            # res_code = -1
            # if os.environ.get('ENCORD_ACTIVE_USE_SUBPROC_ALEMBIC', '0') == '1':
            #     # optional mode that doesn't change the cwd.
            #     # not used by default.
            #     import subprocess
            #     res_code = subprocess.Popen(alembic_args, cwd=alembic_cwd).wait()
            # if res_code != 0:
            # Run via CWD switch.
            current_cwd = os.getcwd()
            try:
                os.chdir(alembic_cwd)
                import alembic.config

                alembic.config.main(argv=alembic_args[1:])
            finally:
                os.chdir(current_cwd)
        else:
            SQLModel.metadata.create_all(engine)
        _init_metadata.add(path_key)
    return engine
