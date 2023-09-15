import enum
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlmodel import CheckConstraint, SQLModel

from encord_active.db.models.project import Project
from encord_active.db.util.fields import (
    field_datetime,
    field_int,
    field_json_dict,
    field_real,
    field_text,
    field_uuid,
    fk_constraint,
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
