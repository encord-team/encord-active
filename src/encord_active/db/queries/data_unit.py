import uuid

from sqlmodel import select
from sqlmodel.sql.expression import Select, SelectOfScalar

from encord_active.db.models import ProjectDataMetadata, ProjectDataUnitMetadata


def select_data_unit_metadata(
    project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int
) -> "SelectOfScalar[ProjectDataUnitMetadata]":
    return select(ProjectDataUnitMetadata).where(
        ProjectDataUnitMetadata.project_hash == project_hash,
        ProjectDataUnitMetadata.data_hash == du_hash,
        ProjectDataUnitMetadata.frame == frame,
    )


def select_data_metadata(
    project_hash: uuid.UUID,
    data_hash: uuid.UUID,
) -> "SelectOfScalar[ProjectDataMetadata]":
    """
    Note: Only use this if you need both `classifictaion_answers` and
    `label_row_json` which are JSON fields and can be huge
    """
    return select(ProjectDataMetadata).where(
        ProjectDataMetadata.project_hash == project_hash,
        ProjectDataMetadata.data_hash == data_hash,
    )


# TODO: find a better way to do this, maybe multiple models for the same table?
def select_data_label_title_fps_answers(
    project_hash: uuid.UUID,
    data_hash: uuid.UUID,
) -> "Select[Tuple[uuid.UUID, str, Optional[float], dict]]":
    return select(
        ProjectDataMetadata.label_hash,
        ProjectDataMetadata.data_title,
        ProjectDataMetadata.frames_per_second,
        ProjectDataMetadata.classifictaion_answers,
    ).where(
        ProjectDataMetadata.project_hash == project_hash,
        ProjectDataMetadata.data_hash == data_hash,
    )
