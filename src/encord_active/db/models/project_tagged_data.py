from uuid import UUID

from sqlmodel import CheckConstraint, SQLModel

from encord_active.db.models.project_data_analytics import ProjectDataAnalytics
from encord_active.db.models.project_tag import ProjectTag
from encord_active.db.util.fields import field_int, field_uuid, fk_constraint


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
