import enum
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from uuid import UUID

from sqlalchemy import INT, JSON, NCHAR, REAL, SMALLINT, CheckConstraint, Column
from sqlalchemy.engine import Engine
from sqlmodel import Field, ForeignKeyConstraint, Index, SQLModel, create_engine

from encord_active.db.enums import AnnotationType, AnnotationTypeMaxValue
from encord_active.db.metrics import (
    AnnotationMetrics,
    DataMetrics,
    MetricDefinition,
    MetricType,
    assert_cls_metrics_match,
)


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


"""
Number of custom metrics supported for each table
"""
CUSTOM_METRIC_COUNT: int = 4


class EmbeddingReductionType(enum.Enum):
    UMAP = "umap"


class Project(SQLModel, table=True):
    __tablename__ = "active_project"
    project_hash: UUID = Field(primary_key=True)
    project_name: str
    project_description: str
    project_remote_ssh_key_path: Optional[str] = Field(nullable=True)
    project_ontology: dict = Field(sa_column=Column(JSON))

    # Custom metadata, list of all metadata for custom metrics
    custom_metrics: list = Field(sa_column=Column(JSON), default=[])


class ProjectImportMetadata(SQLModel, table=True):
    __tablename__ = "active_project_import_metadata"
    project_hash: UUID = Field(primary_key=True)
    import_metadata_type: str
    import_metadata: dict = Field(sa_column=Column(JSON))

    __table_args__ = (fk_constraint(["project_hash"], Project, "active_project_hash_import_meta_fk"),)


class ProjectCollaborator(SQLModel, table=True):
    __tablename__ = "active_project_collaborator"
    project_hash: UUID = Field(primary_key=True)
    user_hash: UUID = Field(primary_key=True)
    user_email: str = Field(default="")


class ProjectEmbeddingReduction(SQLModel, table=True):
    __tablename__ = "active_project_embedding_reduction"
    reduction_hash: UUID = Field(primary_key=True)
    reduction_name: str
    reduction_description: str
    project_hash: UUID

    # Binary encoded information on the 2d reduction implementation.
    reduction_type: EmbeddingReductionType
    reduction_bytes: bytes

    __table_args__ = (fk_constraint(["project_hash"], Project, "active_project_embedding_reduction_project_fk"),)


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
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    data_hash: UUID

    # Size metadata - used for efficient computation scheduling.
    width: int
    height: int

    # Optionally set to support local data (video can be stored as decoded frames or as a video object)
    data_uri: Optional[str] = Field(default=None)
    data_uri_is_video: bool = Field(default=False)

    # Per-frame information about the root cause.
    objects: list = Field(sa_column=Column(JSON))
    classifications: list = Field(sa_column=Column(JSON))

    __table_args__ = (
        Index("active_project_data_units_unique_du_hash_frame", "project_hash", "du_hash", "frame", unique=True),
        fk_constraint(["project_hash", "data_hash"], ProjectDataMetadata, "active_data_unit_data_fk"),
    )


def metric_field_type_normal(nullable: bool = True, override_min: int = 0) -> float:
    return Field(
        # ge=override_min,
        # le=0,
        sa_column=Column(REAL, nullable=nullable),
    )


def metric_field_type_positive_integer(nullable: bool = True) -> int:
    return Field(
        # ge=0,
        sa_column=Column(INT, nullable=nullable)
    )


def metric_field_type_positive_float(nullable: bool = True) -> float:
    return Field(
        # ge=0,
        sa_column=Column(REAL, nullable=nullable)
    )


def field_type_char8(primary_key: bool = False, nullable: bool = False) -> str:
    return Field(
        primary_key=primary_key,
        sa_column=Column(NCHAR(length=8), nullable=nullable, primary_key=primary_key),
        min_length=8,
        max_length=8,
    )


def field_type_annotation_type() -> AnnotationType:
    return Field(sa_column=Column(SMALLINT, nullable=False), ge=0, le=AnnotationTypeMaxValue)


def define_metric_indices(
    metric_prefix: str,
    metrics: Dict[str, MetricDefinition],
    extra: Iterable[Union[Index, CheckConstraint, ForeignKeyConstraint]],
    add_constraints: bool = True,
) -> Tuple[Union[Index, CheckConstraint, ForeignKeyConstraint], ...]:
    values: List[Union[Index, CheckConstraint, ForeignKeyConstraint]] = [
        Index(f"{metric_prefix}_ph_{metric_name}_index", "project_hash", metric_name)
        for metric_name, metric_metadata in metrics.items()
    ]
    if add_constraints:
        for metric_name, metric_metadata in metrics.items():
            name = f"{metric_prefix}_{metric_name}"
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
    __tablename__ = "active_project_analytics_data"
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    # Metrics - Absolute Size
    metric_width: Optional[int] = metric_field_type_positive_integer()
    metric_height: Optional[int] = metric_field_type_positive_integer()
    metric_area: Optional[int] = metric_field_type_positive_integer()
    # Metrics - Relative Size
    metric_aspect_ratio: Optional[float] = metric_field_type_positive_float()
    # Metrics Color
    metric_brightness: Optional[float] = metric_field_type_normal()
    metric_contrast: Optional[float] = metric_field_type_normal()
    metric_sharpness: Optional[float] = metric_field_type_normal()
    metric_red: Optional[float] = metric_field_type_normal()
    metric_green: Optional[float] = metric_field_type_normal()
    metric_blue: Optional[float] = metric_field_type_normal()
    # Random
    metric_random: Optional[float] = metric_field_type_normal()
    # Image Only
    metric_object_count: Optional[int] = metric_field_type_positive_integer()
    metric_object_density: Optional[float] = metric_field_type_normal()
    metric_image_difficulty: Optional[float]  # FIXME: is the output of this always an integer??
    metric_image_uniqueness: Optional[float] = metric_field_type_normal()

    # 4x custom normal metrics
    metric_custom0: Optional[float] = metric_field_type_normal()
    metric_custom1: Optional[float] = metric_field_type_normal()
    metric_custom2: Optional[float] = metric_field_type_normal()
    metric_custom3: Optional[float] = metric_field_type_normal()

    __table_args__ = define_metric_indices(
        "active_data",
        DataMetrics,
        [
            fk_constraint(
                ["project_hash", "du_hash", "frame"], ProjectDataUnitMetadata, "active_data_project_analytics_data_fk"
            )
        ],
    )


class ProjectDataAnalyticsExtra(SQLModel, table=True):
    __tablename__ = "active_project_analytics_data_extra"
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    # Embeddings
    embedding_clip: Optional[bytes]
    # Embeddings derived
    derived_clip_nearest: Optional[dict] = Field(sa_column=Column(JSON))
    # Metric comments
    metric_metadata: dict = Field(sa_column=Column(JSON))

    __table_args__ = (
        fk_constraint(
            ["project_hash", "du_hash", "frame"], ProjectDataAnalytics, "active_data_project_data_analytics_extra_fk"
        ),
    )


class ProjectDataAnalyticsReduced(SQLModel, table=True):
    __tablename__ = "active_project_analytics_data_reduced"
    # Base primary key
    reduction_hash: UUID = Field(primary_key=True)
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    x: float = Field(sa_column=Column(REAL, nullable=False))
    y: float = Field(sa_column=Column(REAL, nullable=False))

    __table_args__ = (
        fk_constraint(
            ["project_hash", "du_hash", "frame"], ProjectDataAnalytics, "active_data_project_data_analytics_reduced_fk"
        ),
        fk_constraint(
            ["reduction_hash"], ProjectEmbeddingReduction, "active_data_project_data_analytics_reduced_reduction_fk"
        ),
        Index("active_project_analytics_data_reduced_x", "reduction_hash", "project_hash", "x", "y"),
        Index("active_project_analytics_data_reduced_y", "reduction_hash", "project_hash", "y", "x"),
    )


@assert_cls_metrics_match(AnnotationMetrics, CUSTOM_METRIC_COUNT)
class ProjectAnnotationAnalytics(SQLModel, table=True):
    __tablename__ = "active_project_analytics_annotation"
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = field_type_char8(primary_key=True)
    # Extra properties
    feature_hash: str = field_type_char8()
    annotation_type: AnnotationType = field_type_annotation_type()
    annotation_user: Optional[UUID]
    annotation_manual: bool
    # Metrics - Absolute Size
    metric_width: Optional[int] = metric_field_type_positive_integer()
    metric_height: Optional[int] = metric_field_type_positive_integer()
    metric_area: Optional[int] = metric_field_type_positive_integer()
    # Metrics - Relative Size
    metric_area_relative: Optional[float] = metric_field_type_normal()
    metric_aspect_ratio: Optional[float] = metric_field_type_positive_float()
    # Metrics Color
    metric_brightness: Optional[float] = metric_field_type_normal()
    metric_contrast: Optional[float] = metric_field_type_normal()
    metric_sharpness: Optional[float] = metric_field_type_normal()
    metric_red: Optional[float] = metric_field_type_normal()
    metric_green: Optional[float] = metric_field_type_normal()
    metric_blue: Optional[float] = metric_field_type_normal()
    # Random
    metric_random: Optional[float] = metric_field_type_normal()
    # Both - Annotation based
    metric_annotation_quality: Optional[float] = metric_field_type_normal()
    # Metrics - Label Only
    metric_max_iou: Optional[float] = metric_field_type_normal()
    metric_border_relative: Optional[float] = metric_field_type_normal()
    metric_label_poly_similarity: Optional[float] = metric_field_type_normal()
    metric_missing_or_broken_track: Optional[float] = metric_field_type_normal()
    metric_inconsistent_class: Optional[float] = metric_field_type_normal()
    metric_label_shape_outlier: Optional[float] = metric_field_type_normal()
    metric_confidence: float = metric_field_type_normal()

    # 4x custom normal metrics
    metric_custom0: Optional[float] = metric_field_type_normal()
    metric_custom1: Optional[float] = metric_field_type_normal()
    metric_custom2: Optional[float] = metric_field_type_normal()
    metric_custom3: Optional[float] = metric_field_type_normal()

    __table_args__ = define_metric_indices(
        "active_annotate",
        AnnotationMetrics,
        [
            fk_constraint(
                ["project_hash", "du_hash", "frame"], ProjectDataUnitMetadata, "active_label_project_data_fk"
            ),
            fk_constraint(
                ["project_hash", "annotation_user"],
                ProjectCollaborator,
                "active_project_label_annotation_user_fk",
                ["project_hash", "user_hash"],
            ),
        ],
    )


class ProjectAnnotationAnalyticsExtra(SQLModel, table=True):
    __tablename__ = "active_project_analytics_annotation_extra"
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = field_type_char8(primary_key=True)
    # Embeddings
    embedding_clip: Optional[bytes]
    embedding_hu: Optional[bytes]
    # Embeddings derived
    derived_clip_nearest: Optional[dict] = Field(sa_column=Column(JSON))
    # Metric comments
    metric_metadata: dict = Field(sa_column=Column(JSON))

    __table_args__ = (
        fk_constraint(
            ["project_hash", "du_hash", "frame", "object_hash"],
            ProjectAnnotationAnalytics,
            "active_data_project_annotation_analytics_extra_fk",
        ),
    )


class ProjectAnnotationAnalyticsReduced(SQLModel, table=True):
    __tablename__ = "active_project_analytics_annotation_reduced"
    # Base primary key
    reduction_hash: UUID = Field(primary_key=True)
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = field_type_char8(primary_key=True)
    # 2D embedding
    x: float = Field(sa_column=Column(REAL, nullable=False))
    y: float = Field(sa_column=Column(REAL, nullable=False))

    __table_args__ = (
        fk_constraint(
            ["project_hash", "du_hash", "frame", "object_hash"],
            ProjectAnnotationAnalytics,
            "active_data_project_annotation_analytics_reduced_fk",
        ),
        fk_constraint(
            ["reduction_hash"],
            ProjectEmbeddingReduction,
            "active_data_project_annotation_analytics_reduced_reduction_fk",
        ),
        Index("active_project_analytics_annotation_reduced_x", "reduction_hash", "project_hash", "x", "y"),
        Index("active_project_analytics_annotation_reduced_y", "reduction_hash", "project_hash", "y", "x"),
    )


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
    object_hash: str = field_type_char8(primary_key=True)
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


@assert_cls_metrics_match(AnnotationMetrics, CUSTOM_METRIC_COUNT)
class ProjectPredictionAnalytics(SQLModel, table=True):
    __tablename__ = "active_project_prediction_analytics"
    prediction_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = field_type_char8(primary_key=True)
    feature_hash: str = field_type_char8()
    project_hash: UUID

    # Prediction data for display.
    annotation_type: AnnotationType = field_type_annotation_type()

    # Prediction metadata: (for iou dependent TP vs FP split & feature hash grouping)
    match_object_hash: Optional[str] = field_type_char8(nullable=True)
    match_feature_hash: Optional[str] = field_type_char8(nullable=True)
    match_duplicate_iou: float = metric_field_type_normal(nullable=False, override_min=-1)  # -1 is always TP
    iou: float = metric_field_type_normal(nullable=False)

    # Prediction object metrics
    # Metrics - Absolute Size
    metric_width: Optional[int] = metric_field_type_positive_integer()
    metric_height: Optional[int] = metric_field_type_positive_integer()
    metric_area: Optional[int] = metric_field_type_positive_integer()
    # Metrics - Relative Size
    metric_area_relative: Optional[float] = metric_field_type_normal()
    metric_aspect_ratio: Optional[float] = metric_field_type_positive_float()
    # Metrics Color
    metric_brightness: Optional[float] = metric_field_type_normal()
    metric_contrast: Optional[float] = metric_field_type_normal()
    metric_sharpness: Optional[float] = metric_field_type_normal()
    metric_red: Optional[float] = metric_field_type_normal()
    metric_green: Optional[float] = metric_field_type_normal()
    metric_blue: Optional[float] = metric_field_type_normal()
    # Random
    metric_random: Optional[float] = metric_field_type_normal()
    # Both - Annotation based
    metric_annotation_quality: Optional[float] = metric_field_type_normal()
    # Metrics - Label Only
    metric_max_iou: Optional[float] = metric_field_type_normal()
    metric_border_relative: Optional[float] = metric_field_type_normal()
    metric_label_poly_similarity: Optional[float] = metric_field_type_normal()
    metric_missing_or_broken_track: Optional[float] = metric_field_type_normal()
    metric_inconsistent_class: Optional[float] = metric_field_type_normal()
    metric_label_shape_outlier: Optional[float] = metric_field_type_normal()
    metric_confidence: float = metric_field_type_normal()

    # 4x custom normal metrics
    metric_custom0: Optional[float] = metric_field_type_normal()
    metric_custom1: Optional[float] = metric_field_type_normal()
    metric_custom2: Optional[float] = metric_field_type_normal()
    metric_custom3: Optional[float] = metric_field_type_normal()

    __table_args__ = (
        fk_constraint(["prediction_hash"], ProjectPrediction, "active_project_prediction_objects_prediction_fk"),
        Index(
            "active_project_prediction_objects_confidence_index",
            "prediction_hash",
            "metric_confidence",
        ),
        Index(
            "active_project_prediction_objects_feature_confidence_index",
            "prediction_hash",
            "feature_hash",
            "metric_confidence",
        ),
        CheckConstraint("iou BETWEEN 0.0 AND 1.0", name="active_project_prediction_iou"),
        CheckConstraint(
            "match_duplicate_iou BETWEEN 0.0 AND 1.0 OR match_duplicate_iou = -1.0::real",
            name="active_project_prediction_match_duplicate_iou",
        ),
    )


class ProjectPredictionAnalyticsExtra(SQLModel, table=True):
    __tablename__ = "active_project_prediction_analytics_extra"
    prediction_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = field_type_char8(primary_key=True)

    # Packed prediction bytes, format depends on annotation_type on associated
    # non-extra table.
    # BB = [x1, y1, x2, y2] (np.float)
    # RBB = [x1, y1, x2, y2, theta] (np.float)
    # BITMASK = count array (np.int)
    # Polygon = [[x, y], [x,y]] (np.float)
    annotation_bytes: bytes

    # Embeddings
    embedding_clip: Optional[bytes]
    embedding_hu: Optional[bytes]
    # Embeddings derived
    derived_clip_nearest: Optional[dict] = Field(sa_column=Column(JSON))
    # Metric comments
    metric_metadata: dict = Field(sa_column=Column(JSON))

    __table_args__ = (
        fk_constraint(
            ["prediction_hash", "du_hash", "frame", "object_hash"],
            ProjectPredictionAnalytics,
            "active_project_prediction_analytics_extra_fk",
        ),
    )


class ProjectPredictionAnalyticsReduced(SQLModel, table=True):
    __tablename__ = "active_project_prediction_analytics_reduced"
    # Base primary key
    reduction_hash: UUID = Field(primary_key=True)
    prediction_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = field_type_char8(primary_key=True)
    # 2D embedding
    x: float = Field(sa_column=Column(REAL, nullable=False))
    y: float = Field(sa_column=Column(REAL, nullable=False))

    __table_args__ = (
        fk_constraint(
            ["prediction_hash", "du_hash", "frame", "object_hash"],
            ProjectPredictionAnalytics,
            "active_data_project_prediction_analytics_reduced_fk",
        ),
        fk_constraint(
            ["reduction_hash"],
            ProjectEmbeddingReduction,
            "active_data_project_prediction_analytics_reduced_reduction_fk",
        ),
        Index("active_project_analytics_prediction_reduced_x", "reduction_hash", "prediction_hash", "x", "y"),
        Index("active_project_analytics_prediction_reduced_y", "reduction_hash", "prediction_hash", "y", "x"),
    )


class ProjectPredictionAnalyticsFalseNegatives(SQLModel, table=True):
    __tablename__ = "active_project_prediction_analytics_false_negatives"
    prediction_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = field_type_char8(primary_key=True)

    # IOU threshold for missed prediction
    # this entry is a false negative IFF (iou < iou_threshold)
    # -1.0 is used for unconditional
    iou_threshold: float = metric_field_type_normal(nullable=False, override_min=-1)

    # Associated feature hash - used for common queries so split out explicitly.
    feature_hash: str = field_type_char8(primary_key=True)

    __table_args__ = (
        fk_constraint(["prediction_hash"], ProjectPrediction, "active_project_prediction_unmatched_prediction_fk"),
        Index("active_project_prediction_unmatched_feature_hash_index", "prediction_hash", "feature_hash"),
        CheckConstraint(
            "iou_threshold BETWEEN 0.0 AND 1.0 OR iou_threshold = -1.0",
            name="active_project_prediction_unmatched_iou_threshold",
        ),
    )


# FIXME: metrics should be inline predictions or separate (need same keys for easy comparison)??


_init_metadata: Set[str] = set()


def get_engine(
    path: Path,
    concurrent: bool = False,
    use_alembic: bool = True,
) -> Engine:
    override_db = os.environ.get("ENCORD_ACTIVE_DATABASE", None)
    create_db_schema = os.environ.get("ENCORD_ACTIVE_DATABASE_SCHEMA_UPDATE", "1")

    path = path.expanduser().resolve()
    engine_url = override_db if override_db is not None else f"sqlite:///{path}"
    connect_args = {"check_same_thread": False} if concurrent and engine_url.startswith("sqlite:/") else {}

    # Create the engine connection
    print(f"Connection to database: {engine_url}")
    engine = create_engine(engine_url, connect_args=connect_args)
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
