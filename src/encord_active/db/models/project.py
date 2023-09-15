from uuid import UUID

from sqlmodel import SQLModel

from encord_active.db.util.fields import (
    field_bool,
    field_json_dict,
    field_json_list,
    field_text,
    field_uuid,
)


class Project(SQLModel, table=True):
    __tablename__ = "project"
    project_hash: UUID = field_uuid(primary_key=True)
    name: str = field_text()
    description: str = field_text()
    ontology: dict = field_json_dict()
    remote: bool = field_bool()

    # Custom metadata, list of all metadata for custom metrics
    custom_metrics: list = field_json_list(default=[])
