from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.analysis.config import create_analysis, default_torch_device
from encord_active.analysis.executor import SimpleExecutor
from encord_active.db.models import (
    Project,
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsExtra,
    ProjectCollaborator,
    ProjectDataAnalytics,
    ProjectDataAnalyticsExtra,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectImportMetadata,
    ProjectPrediction,
)


@dataclass
class ProjectImportSpec:
    project: Project
    project_import_meta: Optional[ProjectImportMetadata]
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
        [],
        database_dir,
        project_hash,
        project.project.project_remote_ssh_key_path,
    )
    data_analytics, data_analytics_extra, annotation_analytics, annotation_analytics_extra, new_collaborators = res

    # Execute embedding reduction.

    # Populate the database.
    with Session(engine) as sess:
        sess.add(project.project)
        if project.project_import_meta is not None:
            sess.add(project.project_import_meta)
        sess.add_all(project.project_data_list)
        sess.add_all(project.project_du_list)
        sess.add_all(new_collaborators)
        sess.commit()
        sess.add_all(data_analytics)
        sess.add_all(data_analytics_extra)
        sess.add_all(annotation_analytics)
        sess.add_all(annotation_analytics_extra)
        sess.commit()


def refresh_project(
    engine: Engine,
    database_dir: Path,
    upsert: ProjectImportSpec,
    force: bool = False,
) -> bool:
    # First identify what values have changed
    project_hash = upsert.project.project_hash
    updated_data_meta = upsert.project_data_list
    updated_du_meta = upsert.project_du_list
    updated_du_set = {(du.du_hash, du.frame) for du in updated_du_meta}
    updated_du_map = {(du.du_hash, du.frame): du for du in updated_du_meta}
    to_update_old: List[ProjectDataUnitMetadata] = []
    to_update: List[ProjectDataUnitMetadata] = []
    to_delete: List[ProjectDataUnitMetadata] = []
    with Session(engine) as sess:
        existing_project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if existing_project is None:
            raise RuntimeError("Project does not exist to be refreshed, please try importing the project instead.")
        predictions = sess.exec(
            select(ProjectPrediction).where(ProjectPrediction.project_hash == project_hash)
        ).fetchall()
        if len(predictions) > 0:
            raise RuntimeError(
                "Project refresh is currently not supported for projects containing predictions, delete predictions to"
                "continue with this operation."
            )
        collaborators = sess.exec(
            select(ProjectCollaborator).where(ProjectCollaborator.project_hash == project_hash)
        ).fetchall()
        existing_du_meta: List[ProjectDataUnitMetadata] = sess.exec(
            select(ProjectDataUnitMetadata).where(ProjectDataUnitMetadata.project_hash == project_hash)
        ).fetchall()
        for du in existing_du_meta:
            # Check if deleted
            if (du.du_hash, du.frame) not in updated_du_map:
                if (du.du_hash, du.frame) in updated_du_set:
                    raise ValueError(f"Duplicate input data unit: {(du.du_hash, du.frame)}")
                to_delete.append(du)
                continue
            # Otherwise check if updated (& keep local storage values the same if used).
            updated_du = updated_du_map.pop((du.du_hash, du.frame))
            updated_du.data_uri = du.data_uri
            updated_du.data_uri_is_video = du.data_uri_is_video
            if force or any(
                getattr(du, f) != getattr(updated_du, f)
                for f in ProjectDataUnitMetadata.__fields__.keys()
                if f not in ["data_uri", "data_uri_is_video"]
                # FIXME: ensure the condition covers all cases where the values could change!
            ):
                to_update.append(updated_du)
                to_update_old.append(du)
        to_add: List[ProjectDataUnitMetadata] = list(updated_du_map.values())

        if len(to_add) == 0 and len(to_update) == 0 and len(to_delete) == 0:
            # No changes.
            return False

        # FIXME: use-smarter logic that re-uses data units metric stage 1 values to make
        # project refreshes faster
        # The following logic just hard-resets everything.

        # Load all metrics that should be 'retained' (skip stage 1 via re-constructing the intermediate state).
        existing_annotation_analytics: List[ProjectAnnotationAnalytics] = sess.exec(
            select(ProjectAnnotationAnalytics).where(ProjectAnnotationAnalytics.project_hash == project_hash)
        ).fetchall()
        existing_annotation_analytics_extra: List[ProjectAnnotationAnalyticsExtra] = sess.exec(
            select(ProjectAnnotationAnalyticsExtra).where(ProjectAnnotationAnalyticsExtra.project_hash == project_hash)
        ).fetchall()
        existing_data_analytics: List[ProjectDataAnalytics] = sess.exec(
            select(ProjectDataAnalytics).where(ProjectDataAnalytics.project_hash == project_hash)
        ).fetchall()
        existing_data_analytics_extra: List[ProjectDataAnalyticsExtra] = sess.exec(
            select(ProjectDataAnalyticsExtra).where(ProjectDataAnalyticsExtra.project_hash == project_hash)
        ).fetchall()
        existing_data_meta: List[ProjectDataMetadata] = sess.exec(
            select(ProjectDataMetadata).where(ProjectDataMetadata.project_hash == project_hash)
        ).fetchall()

    # Create metric engine
    metric_engine = SimpleExecutor(create_analysis(default_torch_device()))

    # FIXME: currently we re-calculate stage 1 metrics for un-changed values
    res = metric_engine.execute_from_db(
        data_meta=updated_data_meta,
        du_meta=updated_du_meta,
        collaborators=collaborators,
        database_dir=database_dir,
        project_hash=project_hash,
        project_ssh_path=upsert.project.project_remote_ssh_key_path,
    )
    data_analysis_new, data_analysis_new_extra, annotate_analysis_new, annotate_analysis_new_extra, collab_new = res

    with Session(engine) as sess:
        # Sync collab & project metadata
        sess.add_all(collab_new)
        existing_project.project_ontology = upsert.project.project_ontology
        existing_project.project_name = upsert.project.project_name
        existing_project.project_description = upsert.project.project_description
        sess.add(existing_project)
        sess.commit()

        # Delete & re-add data units and analytics
        del_list_list: List[
            Union[
                List[ProjectDataMetadata],
                List[ProjectDataUnitMetadata],
                List[ProjectDataAnalytics],
                List[ProjectDataAnalyticsExtra],
                List[ProjectAnnotationAnalytics],
                List[ProjectAnnotationAnalyticsExtra],
            ]
        ] = [
            existing_du_meta,
            existing_data_meta,
            existing_data_analytics,
            existing_data_analytics_extra,
            existing_annotation_analytics,
            existing_annotation_analytics_extra,
        ]
        for del_list in del_list_list:
            for d in del_list:
                sess.delete(d)
        sess.commit()
        sess.add_all(updated_data_meta)
        sess.add_all(updated_du_meta)
        sess.add_all(data_analysis_new)
        sess.add_all(data_analysis_new_extra)
        sess.add_all(annotate_analysis_new)
        sess.add_all(annotate_analysis_new_extra)
    return True
