import enum
from uuid import UUID

from sqlmodel import Index, SQLModel

from encord_active.db.models.project import Project
from encord_active.db.util.fields import field_text, field_uuid, fk_constraint


class EmbeddingReductionType(enum.Enum):
    UMAP = "umap"
    PCA = "pca"


class ProjectEmbeddingReduction(SQLModel, table=True):
    __tablename__ = "project_embedding_reduction"
    reduction_hash: UUID = field_uuid(primary_key=True)
    project_hash: UUID = field_uuid()
    reduction_name: str = field_text()
    reduction_description: str = field_text()

    # Binary encoded information on the 2d reduction implementation.
    reduction_type: EmbeddingReductionType
    reduction_bytes: bytes

    __table_args__ = (
        fk_constraint(["project_hash"], Project, "fk_project_embedding_reduction"),
        Index("ix_project_embedding_reduction", "project_hash", "reduction_hash", unique=True),
    )
