from typing import Optional
from uuid import UUID

from sqlmodel import CheckConstraint, SQLModel

from encord_active.db.models.project_data_metadata import ProjectDataMetadata
from encord_active.db.util.fields import (
    field_annotation_json_list,
    field_bool,
    field_int,
    field_text,
    field_uuid,
    fk_constraint,
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
    objects: list[dict] = field_annotation_json_list()
    classifications: list[dict] = field_annotation_json_list()

    __table_args__ = (
        fk_constraint(["project_hash", "data_hash"], ProjectDataMetadata, "fk_project_data_units"),
        CheckConstraint("frame >= 0", name="project_data_units_frame"),
    )
