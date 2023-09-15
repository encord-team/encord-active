from typing import Optional
from uuid import UUID

from sqlmodel import SQLModel

from encord_active.db.models.project import Project
from encord_active.db.util.fields import (
    field_bool,
    field_bytes,
    field_text,
    field_uuid,
    fk_constraint,
)


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
