import enum
from uuid import UUID

from sqlmodel import SQLModel

from encord_active.db.models.project import Project
from encord_active.db.util.fields import field_json_dict, field_uuid, fk_constraint


class ImportMetadataType(enum.Enum):
    COCO = "coco"


class ProjectImportMetadata(SQLModel, table=True):
    __tablename__ = "project_import"
    project_hash: UUID = field_uuid(primary_key=True)
    import_metadata: dict = field_json_dict()
    import_metadata_type: ImportMetadataType

    __table_args__ = (fk_constraint(["project_hash"], Project, "fk_project_import"),)
