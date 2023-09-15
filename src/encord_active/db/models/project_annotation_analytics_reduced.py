from uuid import UUID

from sqlmodel import CheckConstraint, Index, SQLModel

from encord_active.db.models.project_annotation_analytics import (
    ProjectAnnotationAnalytics,
)
from encord_active.db.models.project_embedding_reduction import (
    ProjectEmbeddingReduction,
)
from encord_active.db.util.fields import (
    field_char8,
    field_int,
    field_real,
    field_uuid,
    fk_constraint,
)


class ProjectAnnotationAnalyticsReduced(SQLModel, table=True):
    __tablename__ = "project_analytics_annotation_reduced"
    # Base primary key
    reduction_hash: UUID = field_uuid(primary_key=True)
    project_hash: UUID = field_uuid(primary_key=True)
    du_hash: UUID = field_uuid(primary_key=True)
    frame: int = field_int(primary_key=True)
    annotation_hash: str = field_char8(primary_key=True)
    # 2D embedding
    x: float = field_real()
    y: float = field_real()

    __table_args__ = (
        fk_constraint(
            ["project_hash", "du_hash", "frame", "annotation_hash"],
            ProjectAnnotationAnalytics,
            "fk_project_analytics_annotation_reduced",
        ),
        fk_constraint(
            ["reduction_hash"],
            ProjectEmbeddingReduction,
            "fk_project_analytics_annotation_reduced_reduction",
        ),
        Index("ix_project_analytics_annotation_reduced_x", "reduction_hash", "project_hash", "x", "y"),
        Index("ix_project_analytics_annotation_reduced_y", "reduction_hash", "project_hash", "y", "x"),
        CheckConstraint("frame >= 0", name="project_analytics_annotation_reduced_frame"),
    )
