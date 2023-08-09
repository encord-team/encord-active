from dataclasses import dataclass
from pathlib import Path
from typing import List

from sqlalchemy.engine import Engine
from sqlmodel import Session

from encord_active.analysis.config import create_analysis, default_torch_device
from encord_active.analysis.executor import SimpleExecutor
from encord_active.db.models import (
    Project,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
)


@dataclass
class ProjectImportSpec:
    project: Project
    project_data_list: List[ProjectDataMetadata]
    project_du_list: List[ProjectDataUnitMetadata]


def import_project(engine: Engine, database_dir: Path, project: ProjectImportSpec) -> None:
    """
    Imports a project-spec into active.
    """
    # Validation
    project_hash = project.project.project_hash
    for data in project.project_data_list:
        if data.project_hash != project_hash:
            raise ValueError(f"Project import, project hash inconsistency: {data.project_hash} != {project_hash}")
    for du in project.project_du_list:
        if du.project_hash != project_hash:
            raise ValueError(f"Project import, project hash inconsistency: {du.project_hash} != {project_hash}")

    # Execute metric engine.
    metric_engine = SimpleExecutor(create_analysis(default_torch_device()))
    res = metric_engine.execute_from_db(
        project.project_data_list,
        project.project_du_list,
        database_dir,
        project_hash,
        project.project.project_remote_ssh_key_path,
    )
    data_analytics, data_analytics_extra, annotation_analytics, annotation_analytics_extra = res

    # Execute embedding reduction.

    # Populate the database.
    with Session(engine) as sess:
        sess.add(project.project)
        sess.add_all(project.project_data_list)
        sess.add_all(project.project_du_list)
        sess.commit()
        sess.add_all(data_analytics)
        sess.add_all(data_analytics_extra)
        sess.add_all(annotation_analytics)
        sess.add_all(annotation_analytics_extra)
        sess.commit()


def update_project(
    engine: Engine,
    upsert: ProjectImportSpec,
    delete: None,
    run_stage_2: bool,
) -> None:
    """
    Update a project into active.
    Changes supported:
        - different project name / description
        - new data units
        - new data
        - delete data units
        - delete data
    Option:
        run_stage_2 if set to False will disable the slower stage 2 evaluation
    """
    raise ValueError("Not yet implemented")
