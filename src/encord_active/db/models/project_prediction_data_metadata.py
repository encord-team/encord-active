from datetime import datetime
from uuid import UUID

from sqlmodel import SQLModel

from encord_active.db.models.project_data_metadata import ProjectDataMetadata
from encord_active.db.models.project_prediction import ProjectPrediction
from encord_active.db.util.fields import (
    field_datetime,
    field_json_dict,
    field_uuid,
    fk_constraint,
)


class ProjectPredictionDataMetadata(SQLModel, table=True):
    __tablename__ = "prediction_data"
    # Base primary key
    prediction_hash: UUID = field_uuid(primary_key=True)
    data_hash: UUID = field_uuid(primary_key=True)

    # Associated project_hash
    project_hash: UUID = field_uuid()

    # Metadata
    label_hash: UUID = field_uuid(unique=True)
    created_at: datetime = field_datetime()
    last_edited_at: datetime = field_datetime()
    object_answers: dict = field_json_dict()
    classification_answers: dict = field_json_dict()

    __table_args__ = (
        fk_constraint(["prediction_hash", "project_hash"], ProjectPrediction, "fk_prediction_data"),
        fk_constraint(["project_hash", "data_hash"], ProjectDataMetadata, "fk_prediction_data_project"),
    )
