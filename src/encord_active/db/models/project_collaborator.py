from uuid import UUID

from sqlmodel import SQLModel

from encord_active.db.models.project import Project
from encord_active.db.util.fields import (
    field_big_int,
    field_encrypted_text,
    field_uuid,
    fk_constraint,
)


class ProjectCollaborator(SQLModel, table=True):
    __tablename__ = "project_collaborator"
    project_hash: UUID = field_uuid(primary_key=True)
    user_id: int = field_big_int(primary_key=True)
    user_email: str = field_encrypted_text()

    __table_args__ = (fk_constraint(["project_hash"], Project, "fk_project_collaborator"),)
