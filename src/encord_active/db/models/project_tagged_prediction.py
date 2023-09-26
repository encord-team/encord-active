from uuid import UUID

from sqlmodel import CheckConstraint, SQLModel

from encord_active.db.models import ProjectPrediction
from encord_active.db.models.project_prediction_analytics import (
    ProjectPredictionAnalytics,
)
from encord_active.db.models.project_tag import ProjectTag
from encord_active.db.util.fields import (
    field_char8,
    field_int,
    field_uuid,
    fk_constraint,
)


class ProjectTaggedPrediction(SQLModel, table=True):
    __tablename__ = "project_tagged_prediction"
    prediction_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    tag_hash: UUID = field_uuid(primary_key=True)
    project_hash: UUID = field_uuid()

    __table_args__ = (
        fk_constraint(
            ["prediction_hash", "du_hash", "frame", "annotation_hash"],
            ProjectPredictionAnalytics,
            "fk_project_tagged_prediction",
        ),
        fk_constraint(
            ["project_hash", "prediction_hash"],
            ProjectPrediction,
            "fk_project_tagged_prediction_project",
        ),
        fk_constraint(["tag_hash"], ProjectTag, "fk_project_tagged_prediction_tag"),
        CheckConstraint("frame >= 0", name="project_tagged_prediction_frame"),
    )
