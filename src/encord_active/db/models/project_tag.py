from uuid import UUID

from sqlmodel import Index, SQLModel

from encord_active.db.models.project import Project
from encord_active.db.util.fields import field_text, field_uuid, fk_constraint


class ProjectTag(SQLModel, table=True):
    __tablename__ = "project_tags"
    tag_hash: UUID = field_uuid(primary_key=True)
    project_hash: UUID = field_uuid()
    name: str = field_text()
    description: str = field_text()

    __table_args__ = (
        fk_constraint(["project_hash"], Project, "fk_project_tags"),
        Index("uq_project_tags_name", "project_hash", "name", unique=True),
    )
