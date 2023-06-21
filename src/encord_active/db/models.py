import os
from pathlib import Path
from typing import Optional, Set
from uuid import UUID

from sqlalchemy import Column, JSON
from sqlalchemy.engine import Engine
from sqlmodel import Field, SQLModel, create_engine, Index, Session


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
    num_frames: int
    frames_per_second: float
    dataset_title: str
    data_title: str
    data_type: str
    label_row_json: dict = Field(sa_column=Column(JSON))


class ProjectDataUnitMetadata(SQLModel, table=True):
    __tablename__ = 'active_project_data_units'
    project_hash: UUID = Field(primary_key=True)
    data_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True)
    data_unit_hash: UUID
    __table_args__ = (
        Index("unique_data_unit_hash", "data_unit_hash", "frame", unique=True),
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
    data_hash: UUID = Field(primary_key=True)
    frame: int = Field(primary_key=True)
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


class ProjectDataAnalytics(ProjectAnalyticsBase, table=True):
    __tablename__ = 'active_project_data_analytics'
    # Metrics - Image only
    metric_object_count: Optional[int]
    metric_object_density: Optional[float]
    metric_image_difficulty: Optional[float]
    metric_image_singularity: Optional[float]

    __table_args__ = tuple([
        Index(
            f"active_data_project_hash_{metric_name}_index", "project_hash", metric_name
        )
        for metric_name in (
                {
                    "metric_object_count",
                    "metric_object_density"
                } | COMMON_METRIC_NAMES
        )
    ])


class ProjectLabelAnalytics(ProjectAnalyticsBase, table=True):
    __tablename__ = 'active_project_label_analytics'
    # Extended primary key
    object_hash: str = Field(primary_key=True)
    # Metrics - Label Only
    metric_label_duplicates: Optional[float]
    metric_label_border_closeness: Optional[float]
    metric_label_poly_similarity: Optional[float]
    metric_label_missing_or_broken_tracks: Optional[float]
    metric_label_annotation_quality: Optional[float]
    metric_label_inconsistent_classification_and_track: Optional[float]
    __table_args__ = tuple([
        Index(
            f"active_label_project_hash_{metric_name}_index", "project_hash", metric_name
        )
        for metric_name in (
                {
                    "metric_label_duplicates",
                    "metric_label_border_closeness",
                    "metric_label_poly_similarity",
                    "metric_label_missing_or_broken_tracks",
                    "metric_label_annotation_quality",
                    "metric_label_inconsistent_classification_and_track"
                } | COMMON_METRIC_NAMES
        )
    ])


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
    override_db = os.getenv("ENCORD_ACTIVE_DATABASE", None)
    create_db_schema = os.getenv("ENCORD_ACTIVE_DATABASE_SCHEMA_UPDATE", "1")

    engine = create_engine(override_db if override_db is not None else f"sqlite:///{path}")
    path_key = path.as_posix()
    if path_key not in _init_metadata and create_db_schema == "1":
        SQLModel.metadata.create_all(engine)
        _init_metadata.add(path_key)
    return engine
