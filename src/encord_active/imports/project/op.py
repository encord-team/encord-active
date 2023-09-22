import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Union

import numpy as np
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.analysis.config import create_analysis, default_torch_device
from encord_active.analysis.executor import ExecutionResult, SimpleExecutor
from encord_active.analysis.reductions.op import (
    apply_embedding_reduction,
    create_reduction,
    serialize_reduction,
)
from encord_active.db.models import (
    EmbeddingReductionType,
    Project,
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsExtra,
    ProjectAnnotationAnalyticsReduced,
    ProjectCollaborator,
    ProjectDataAnalytics,
    ProjectDataAnalyticsExtra,
    ProjectDataAnalyticsReduced,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectEmbeddingIndex,
    ProjectImportMetadata,
    ProjectPrediction,
)
from encord_active.db.models.project_embedding_reduction import (
    ProjectEmbeddingReduction,
)


@dataclass
class ProjectImportSpec:
    project: Project
    project_import_meta: Optional[ProjectImportMetadata]
    project_data_list: List[ProjectDataMetadata]
    project_du_list: List[ProjectDataUnitMetadata]

    @property
    def is_empty(self):
        return len(self.project_data_list) + len(self.project_du_list) == 0


class ReductionResult(NamedTuple):
    project_reduction: ProjectEmbeddingReduction
    reduced_data_clip: list[ProjectDataAnalyticsReduced]
    reduced_annotation_clip: list[ProjectAnnotationAnalyticsReduced]
    embedding_index: ProjectEmbeddingIndex


def _reduce_project_embeddings(project: ProjectImportSpec, exec_res: ExecutionResult) -> ReductionResult:
    reduction_total_samples: List[Union[ProjectDataAnalyticsExtra, ProjectAnnotationAnalyticsExtra]] = []
    reduction_total_samples.extend(exec_res.data_analytics_extra)
    reduction_total_samples.extend(exec_res.annotation_analytics_extra)
    reduction_train_samples: List[Union[ProjectDataAnalyticsExtra, ProjectAnnotationAnalyticsExtra]] = random.choices(
        reduction_total_samples, k=min(len(reduction_total_samples), 50_000)
    )
    reduction_train_samples.sort(
        key=lambda x: (x.du_hash, x.frame, x.annotation_hash if isinstance(x, ProjectAnnotationAnalyticsExtra) else "")
    )
    reduction = create_reduction(
        EmbeddingReductionType.UMAP,
        train_samples=[
            np.frombuffer(sample.embedding_clip or b"", dtype=np.float64) for sample in reduction_train_samples
        ],
    )
    project_reduction = serialize_reduction(
        reduction,
        name="Default CLIP",
        description="",
        project_hash=project.project.project_hash,
    )
    reduced_data_clip_raw = apply_embedding_reduction(
        [np.frombuffer(sample.embedding_clip or b"", dtype=np.float64) for sample in exec_res.data_analytics_extra],
        reduction,
    )
    reduced_data_clip = [
        ProjectDataAnalyticsReduced(
            reduction_hash=project_reduction.reduction_hash,
            project_hash=project.project.project_hash,
            du_hash=extra.du_hash,
            frame=extra.frame,
            x=x,
            y=y,
        )
        for (extra, (x, y)) in zip(exec_res.data_analytics_extra, reduced_data_clip_raw)
    ]
    reduced_annotation_clip_raw = apply_embedding_reduction(
        [
            np.frombuffer(sample.embedding_clip or b"", dtype=np.float64)
            for sample in exec_res.annotation_analytics_extra
        ],
        reduction,
    )
    reduced_annotation_clip = [
        ProjectAnnotationAnalyticsReduced(
            reduction_hash=project_reduction.reduction_hash,
            project_hash=project.project.project_hash,
            du_hash=extra.du_hash,
            frame=extra.frame,
            annotation_hash=extra.annotation_hash,
            x=x,
            y=y,
        )
        for (extra, (x, y)) in zip(exec_res.annotation_analytics_extra, reduced_annotation_clip_raw)
    ]

    embedding_index = ProjectEmbeddingIndex(
        project_hash=project.project.project_hash,
        data_index_dirty=True,
        data_index_name=None,
        data_index_compiled=None,
        annotation_index_dirty=True,
        annotation_index_name=None,
        annotation_index_compiled=None,
    )
    return ReductionResult(project_reduction, reduced_data_clip, reduced_annotation_clip, embedding_index)


def import_project(engine: Engine, database_dir: Path, project: ProjectImportSpec, ssh_key: str) -> None:
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
    exec_res = metric_engine.execute_from_db(
        project.project_data_list,
        project.project_du_list,
        [],
        database_dir=database_dir,
        project_hash=project_hash,
        project_ssh_key=ssh_key if project.project.remote else None,
        project_ontology=project.project.ontology,
    )
    # Execute embedding reduction.
    red_res = _reduce_project_embeddings(project, exec_res)

    # Populate the database.
    with Session(engine) as sess:
        with sess.begin():
            sess.bulk_save_objects([project.project])
            if project.project_import_meta is not None:
                sess.add(project.project_import_meta)
            sess.bulk_save_objects(project.project_data_list)
            sess.bulk_save_objects(project.project_du_list)
            sess.bulk_save_objects(exec_res.collaborators)
            sess.bulk_save_objects([red_res.project_reduction])
            sess.bulk_save_objects([red_res.embedding_index])
            sess.add_all(exec_res.data_analytics)
            sess.add_all(exec_res.data_analytics_extra)
            sess.add_all(exec_res.data_analytics_derived)
            sess.add_all(exec_res.annotation_analytics)
            sess.add_all(exec_res.annotation_analytics_extra)
            sess.add_all(exec_res.annotation_analytics_derived)
            sess.add_all(red_res.reduced_data_clip)
            sess.add_all(red_res.reduced_annotation_clip)


def refresh_project(
    engine: Engine,
    database_dir: Path,
    ssh_key: str,
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
    exec_res = metric_engine.execute_from_db(
        data_meta=updated_data_meta,
        du_meta=updated_du_meta,
        collaborators=collaborators,
        database_dir=database_dir,
        project_hash=project_hash,
        project_ssh_key=ssh_key if upsert.project.remote else None,
        project_ontology=upsert.project.ontology,
    )

    with Session(engine) as sess:
        # Sync collab & project metadata
        sess.add_all(exec_res.collaborators)
        existing_project.ontology = upsert.project.ontology
        existing_project.name = upsert.project.name
        existing_project.description = upsert.project.description
        sess.add(existing_project)
        sess.commit()

        # Delete & re-add data units and analytics
        for del_list in [
            existing_du_meta,
            existing_data_meta,
            existing_data_analytics,
            existing_data_analytics_extra,
            existing_annotation_analytics,
            existing_annotation_analytics_extra,
        ]:
            for d in del_list:
                sess.delete(d)
        sess.commit()
        sess.add_all(updated_data_meta)
        sess.add_all(updated_du_meta)
        sess.add_all(exec_res.data_analytics)
        sess.add_all(exec_res.data_analytics_extra)
        sess.add_all(exec_res.annotation_analytics)
        sess.add_all(exec_res.annotation_analytics_extra)
    return True
