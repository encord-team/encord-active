from uuid import UUID

from sqlmodel import CheckConstraint, SQLModel

from encord_active.db.models.project_data_unit_metadata import ProjectDataUnitMetadata
from encord_active.db.models.project_prediction import ProjectPrediction
from encord_active.db.models.project_prediction_data_metadata import (
    ProjectPredictionDataMetadata,
)
from encord_active.db.util.fields import (
    field_annotation_json_list,
    field_int,
    field_uuid,
    fk_constraint,
)


class ProjectPredictionDataUnitMetadata(SQLModel, table=True):
    __tablename__ = "prediction_data_units"
    prediction_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)

    # Associated project_hash / data_hash
    project_hash: UUID = field_uuid()
    data_hash: UUID = field_uuid()

    # Per-frame information about the root cause.
    objects: list = field_annotation_json_list()
    classifications: list = field_annotation_json_list()

    __table_args__ = (
        fk_constraint(
            ["prediction_hash", "data_hash"],
            ProjectPredictionDataMetadata,
            "fk_prediction_data_units",
        ),
        fk_constraint(["project_hash", "du_hash", "frame"], ProjectDataUnitMetadata, "fk_prediction_data_units_data"),
        fk_constraint(["prediction_hash", "project_hash"], ProjectPrediction, "fk_prediction_data_units_project"),
        CheckConstraint("frame >= 0", name="prediction_data_units_frame"),
    )
