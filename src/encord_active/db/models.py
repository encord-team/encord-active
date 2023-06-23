import os
from pathlib import Path
from typing import Optional, Set, Tuple, Union, Iterable
from uuid import UUID

from sqlalchemy import Column, JSON
from sqlalchemy.engine import Engine
from sqlmodel import Field, SQLModel, create_engine, Index, ForeignKeyConstraint, CheckConstraint


class Project(SQLModel, table=True):
    __tablename__ = 'active_project'
    project_hash: UUID = Field(primary_key=True)
    project_name: str
    project_description: str
    project_remote_ssh_key_path: Optional[str] = Field(nullable=True)
    project_ontology: dict = Field(sa_column=Column(JSON))


class ProjectDataMetadata(SQLModel, table=True):
    __tablename__ = 'active_project_data'
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

    __table_args__ = (
        ForeignKeyConstraint(
            ["project_hash"],
            [Project.project_hash],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="active_project_data_project_fk",
        ),
        # CheckConstraint(
        #    """
        #    CHECK (
        #        (data_type = 'video' AND frames_per_second IS NOT NULL AND frames_per_second > 0)
        #        OR (data_type = 'img_group' AND frames_per_second = 1)
        #        OR (data_type = 'image' AND num_frames = 1 AND frames_per_second IS NULL)
        #    )
        #    """,
        #    name="active_project_data_video_consistency"
        # )
    )


class ProjectDataUnitMetadata(SQLModel, table=True):
    __tablename__ = 'active_project_data_units'
    project_hash: UUID = Field(primary_key=True, foreign_key=Project.project_hash)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    data_hash: UUID
    __table_args__ = (
        Index("active_project_data_units_unique_du_hash_frame", "du_hash", "frame", unique=True),
        ForeignKeyConstraint(
            ["project_hash", "data_hash"],
            [ProjectDataMetadata.project_hash, ProjectDataMetadata.data_hash],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="active_data_unit_data_fk",
        ),
    )
    # Optionally set to support local data (video can be stored as decoded frames or as a video object)
    data_uri: Optional[str] = Field(default=None)
    data_uri_is_video: bool = Field(default=False)
    # Per-frame information about the root cause.
    objects: list = Field(sa_column=Column(JSON))
    classifications: list = Field(sa_column=Column(JSON))


class ProjectAnalyticsBase(SQLModel):
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    # Embeddings
    embedding_clip: Optional[bytes]
    embedding_hu: Optional[bytes]
    # Metrics - Absolute Size
    metric_width: Optional[int]
    metric_height: Optional[int]
    metric_area: Optional[int]
    # Metrics - Relative Size
    metric_area_relative: Optional[float]
    metric_aspect_ratio: Optional[float]
    # Metrics Color
    metric_brightness: Optional[float]
    metric_contrast: Optional[float]
    metric_blur: Optional[float]
    metric_red: Optional[float]
    metric_green: Optional[float]
    metric_blue: Optional[float]
    # Random
    metric_random: Optional[float]


COMMON_METRIC_NAMES: Set[str] = {
    field
    for field in ProjectAnalyticsBase.__fields__
    if field.startswith("metric_")
}


def shared_metrics_table_args(
        metric_prefix: str,
        extra_values: Set[str],
        extra_constraints: Iterable[Union[Index, ForeignKeyConstraint]],
        include_common_metrics: bool = True,
) -> Tuple[Union[Index, ForeignKeyConstraint], ...]:
    values = [
        Index(
            f"{metric_prefix}_project_hash_{metric_name}_index", "project_hash", metric_name
        )
        for metric_name in extra_values | (COMMON_METRIC_NAMES if include_common_metrics else set())
    ]
    values = values + list(extra_constraints)
    return tuple(values)


class ProjectDataAnalytics(ProjectAnalyticsBase, table=True):
    __tablename__ = 'active_project_data_analytics'
    # Metrics - Image only
    metric_object_count: Optional[int]
    metric_object_density: Optional[float]
    metric_image_difficulty: Optional[float]
    metric_image_singularity: Optional[float]

    __table_args__ = shared_metrics_table_args(
        "active_data",
        {
            "metric_object_count",
            "metric_object_density"
        },
        [
            ForeignKeyConstraint(
                ["project_hash", "du_hash", "frame"],
                [ProjectDataUnitMetadata.project_hash, ProjectDataUnitMetadata.du_hash,
                 ProjectDataUnitMetadata.frame],
                onupdate="CASCADE",
                ondelete="CASCADE",
                name="active_data_project_data_fk",
            ),
        ]
    )


class ProjectLabelAnalytics(ProjectAnalyticsBase, table=True):
    __tablename__ = 'active_project_label_analytics'
    # Extended primary key
    object_hash: str = Field(primary_key=True, min_length=8, max_length=8)
    feature_hash: str = Field(min_length=8, max_length=8)
    # Metrics - Label Only
    metric_label_duplicates: Optional[float]
    metric_label_border_closeness: Optional[float]
    metric_label_poly_similarity: Optional[float]
    metric_label_missing_or_broken_tracks: Optional[float]
    metric_label_annotation_quality: Optional[float]
    metric_label_inconsistent_classification_and_track: Optional[float]

    __table_args__ = shared_metrics_table_args(
        "active_label",
        {
            "metric_label_duplicates",
            "metric_label_border_closeness",
            "metric_label_poly_similarity",
            "metric_label_missing_or_broken_tracks",
            "metric_label_annotation_quality",
            "metric_label_inconsistent_classification_and_track",
            # Not a metric, but indexed anyway.
            "feature_hash",
        },
        [
            ForeignKeyConstraint(
                ["project_hash", "du_hash", "frame"],
                [ProjectDataUnitMetadata.project_hash, ProjectDataUnitMetadata.du_hash,
                 ProjectDataUnitMetadata.frame],
                onupdate="CASCADE",
                ondelete="CASCADE",
                name="active_label_project_data_fk",
            ),
        ]
    )


class ProjectClassificationAnalytics(SQLModel, table=True):
    __tablename__ = 'active_project_classification_analytics'
    # Base primary key
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    classification_hash: str = Field(primary_key=True, min_length=8, max_length=8)
    feature_hash: str = Field(min_length=8, max_length=8)
    # Classification metrics (most metrics are shared with data metrics)
    # so this only contains metrics that are depend on the classification itself.
    __table_args__ = shared_metrics_table_args(
        "active_classification",
        {
            # Metrics
            # Not a metric, but indexed anyway.
            "feature_hash",
        },
        [
            ForeignKeyConstraint(
                ["project_hash", "du_hash", "frame"],
                [ProjectDataUnitMetadata.project_hash, ProjectDataUnitMetadata.du_hash,
                 ProjectDataUnitMetadata.frame],
                onupdate="CASCADE",
                ondelete="CASCADE",
                name="active_label_project_data_fk",
            ),
        ],
        include_common_metrics=False
    )


class ProjectTag(SQLModel, table=True):
    __tablename__ = 'active_project_tags'
    tag_hash: UUID = Field(primary_key=True)
    project_hash: UUID = Field(index=True)
    name: str = Field(min_length=1, max_length=256)

    __table_args__ = (
        ForeignKeyConstraint(
            ["project_hash"],
            [Project.project_hash],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="active_project_tags_project_fk",
        ),
    )


class ProjectTaggedDataUnit(SQLModel, table=True):
    __tablename__ = 'active_project_tagged_data_units'
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    tag_hash: UUID = Field(primary_key=True, index=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["tag_hash"],
            [ProjectTag.tag_hash],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="active_project_tagged_data_units_tag_fk",
        ),
        ForeignKeyConstraint(
            ["project_hash", "du_hash", "frame"],
            [ProjectDataAnalytics.project_hash, ProjectDataAnalytics.du_hash, ProjectDataAnalytics.frame],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="active_project_tagged_data_units_analysis_fk",
        )
    )


class ProjectTaggedLabel(SQLModel, table=True):
    __tablename__ = 'active_project_tagged_labels'
    project_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = Field(primary_key=True, min_length=8, max_length=8)
    tag_hash: UUID = Field(primary_key=True, index=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["tag_hash"],
            [ProjectTag.tag_hash],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="active_project_tagged_labels_tag_fk",
        ),
        ForeignKeyConstraint(
            ["project_hash", "du_hash", "frame", "object_hash"],
            [ProjectLabelAnalytics.project_hash, ProjectLabelAnalytics.du_hash,
             ProjectLabelAnalytics.frame, ProjectLabelAnalytics.object_hash],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="active_project_tagged_labels_analysis_fk",
        )
    )


class ProjectPrediction(SQLModel, table=True):
    __tablename__ = 'active_project_prediction'
    prediction_hash: UUID = Field(primary_key=True)
    project_hash: UUID
    name: str
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_hash"],
            [Project.project_hash],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="active_project_prediction_project_hash_fk",
        ),
    )


class ProjectPredictionLabelResults(SQLModel, table=True):
    __tablename__ = 'active_project_prediction_label_results'
    prediction_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    object_hash: str = Field(primary_key=True, min_length=8, max_length=8)

    __table_args__ = (
        ForeignKeyConstraint(
            ["prediction_hash"],
            [ProjectPrediction.prediction_hash],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="active_project_prediction_labels_prediction_fk",
        ),
    )


class ProjectPredictionClassificationResults(SQLModel, table=True):
    __tablename__ = 'active_project_prediction_classification_results'
    prediction_hash: UUID = Field(primary_key=True)
    du_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True, ge=0)
    classification_hash: str = Field(primary_key=True, min_length=8, max_length=8)

    __table_args__ = (
        ForeignKeyConstraint(
            ["prediction_hash"],
            [ProjectPrediction.prediction_hash],
            onupdate="CASCADE",
            ondelete="CASCADE",
            name="active_project_prediction_classifications_prediction_fk",
        ),
    )


# FIXME: metrics should be inline predictions or separate (need same keys for easy comparison)


METRICS_DATA = {
    field
    for field in ProjectDataAnalytics.__fields__
    if field.startswith("metric_")
}

EMBEDDINGS_DATA = {
    field
    for field in ProjectDataAnalytics.__fields__
    if field.startswith("embedding_")
}

METRICS_LABEL = {
    field
    for field in ProjectLabelAnalytics.__fields__
    if field.startswith("metric_")
}

EMBEDDINGS_LABEL = {
    field
    for field in ProjectLabelAnalytics.__fields__
    if field.startswith("embedding_")
}

_init_metadata: Set[str] = set()


def get_engine(path: Path) -> Engine:
    override_db = os.environ.get("ENCORD_ACTIVE_DATABASE", None)
    create_db_schema = os.environ.get("ENCORD_ACTIVE_DATABASE_SCHEMA_UPDATE", "1")

    engine = create_engine(override_db if override_db is not None else f"sqlite:///{path}")
    path_key = path.as_posix()
    if path_key not in _init_metadata and create_db_schema == "1":
        # import encord_active.db.migrations.env as migrate_env
        # FIXME: use alembic to auto-run migrations instead of SQLModel!!
        SQLModel.metadata.create_all(engine)
        _init_metadata.add(path_key)
    return engine
