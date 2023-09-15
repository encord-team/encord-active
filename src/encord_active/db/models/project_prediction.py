from typing import Optional
from uuid import UUID

from sqlmodel import Index, SQLModel

from encord_active.db.models.project import Project
from encord_active.db.util.fields import field_uuid, fk_constraint


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
