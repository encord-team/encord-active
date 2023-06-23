from typing import List

from sqlmodel import Session, update, insert

from encord_active.db.models import Project, ProjectDataMetadata, ProjectDataUnitMetadata


def upsert_project_and_data(
    session: Session,
    project: Project,
    data: List[ProjectDataMetadata],
    data_units: List[ProjectDataUnitMetadata]
) -> None:
    session.add(project)
    session.add_all(data)
    session.add_all(data_units)