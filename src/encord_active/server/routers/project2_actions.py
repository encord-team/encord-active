import uuid
from typing import Dict, List, Set, Optional, Type, Union, TypeVar

from encord.orm.project import (
    CopyDatasetAction,
    CopyDatasetOptions,
    CopyLabelsOptions,
    ReviewApprovalState,
)
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, literal
from sqlalchemy.engine import Engine
from sqlalchemy.sql.operators import in_op
from sqlmodel import Session, insert, select

from encord_active.analysis.config import create_analysis, default_torch_device
from encord_active.analysis.executor import SimpleExecutor
from encord_active.db.models import (
    Project,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectDataAnalytics,
    ProjectDataAnalyticsExtra,
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsExtra,
    ProjectCollaborator,
)
from encord_active.lib.encord.utils import get_encord_project
from encord_active.server.dependencies import dep_engine
from encord_active.server.routers.queries import search_query
from encord_active.server.routers.queries.domain_query import TABLES_DATA
from encord_active.server.routers.queries.search_query import SearchFilters
from encord_active.server.settings import get_settings

router = APIRouter(
    prefix="/{project_hash}/actions",
)


class CreateProjectSubsetPostAction(BaseModel):
    project_title: str
    project_description: Optional[str]
    dataset_title: str
    dataset_description: Optional[str]
    filters: SearchFilters


SubsetTableType = TypeVar(
    "SubsetTableType",
    Type[ProjectDataAnalytics],
    Type[ProjectDataAnalyticsExtra],
    Type[ProjectAnnotationAnalytics],
    Type[ProjectAnnotationAnalyticsExtra],
)


@router.post("/create_project_subset")
def route_action_create_project_subset(
    project_hash: uuid.UUID, item: CreateProjectSubsetPostAction, engine: Engine = Depends(dep_engine)
) -> None:
    with Session(engine) as sess:
        current_project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if current_project is None:
            raise ValueError("Unknown project")
        new_local_project_hash = uuid.uuid4()
        new_project = Project(
            project_hash=new_local_project_hash,
            project_name=item.project_title,
            project_description=item.project_description or "",
            project_remote_ssh_key_path=current_project.project_remote_ssh_key_path,
            project_ontology=current_project.project_ontology,
            custom_metrics=current_project.custom_metrics,
        )

        current_collaborators = sess.exec(
            select(ProjectCollaborator).where(ProjectCollaborator.project_hash == project_hash)
        ).fetchall()

        where = search_query.search_filters(
            tables=TABLES_DATA,
            base=TABLES_DATA.primary.analytics,
            search=item.filters,
            project_filters={"project_hash": [project_hash]},
        )

        # Return the complete list of data_hashes (granularity of subset creation that we support).
        data_hashes_query = (
            select(ProjectDataUnitMetadata.data_hash)
            .where(
                *where,
                ProjectDataUnitMetadata.project_hash == project_hash,
                ProjectDataUnitMetadata.du_hash == ProjectDataAnalytics.du_hash,
                ProjectDataUnitMetadata.frame == ProjectDataAnalytics.frame,
            )
            .group_by(ProjectDataUnitMetadata.data_hash)
        )

        # Fetch all analytics state & run stage 2 & 3 re-calculation.
        subset_data_hashes = sess.exec(data_hashes_query).fetchall()

        def _fetch_all(table: Type[SubsetTableType]) -> List[SubsetTableType]:
            return sess.exec(
                select(table).where(
                    table.project_hash == project_hash,
                    ProjectDataUnitMetadata.project_hash == project_hash,
                    table.du_hash == ProjectDataUnitMetadata.du_hash,
                    table.frame == ProjectDataUnitMetadata.frame,
                    in_op(ProjectDataUnitMetadata.data_hash, subset_data_hashes),
                )
            ).fetchall()

        # Fetch all the subset state (used to avoid calculating stage 1 metrics)
        subset_data_analytics = _fetch_all(ProjectDataAnalytics)
        subset_data_analytics_extra = _fetch_all(ProjectDataAnalyticsExtra)
        subset_annotation_analytics = _fetch_all(ProjectAnnotationAnalytics)
        subset_annotation_analytics_extra = _fetch_all(ProjectAnnotationAnalyticsExtra)

        # Use this info to generate NEW values to insert
        metric_engine = SimpleExecutor(create_analysis(default_torch_device()))
        settings = get_settings()
        new_analytics = metric_engine.execute_from_db_subset(
            database_dir=settings.SERVER_START_PATH.expanduser().resolve(),
            project_hash=new_local_project_hash,
            precalculated_data=subset_data_analytics,
            precalculated_data_extra=subset_data_analytics_extra,
            precalculated_annotation=subset_annotation_analytics,
            precalculated_annotation_extra=subset_annotation_analytics_extra,
            precalculated_collaborators=current_collaborators,
        )
        new_data, new_data_extra, new_annotate, new_annotate_extra, new_collab = new_analytics

        # Insert state into the database
        new_local_dataset_hash = uuid.uuid4()
        sess.execute(insert(Project).values(**{k: getattr(new_project, k) for k in Project.__fields__}))
        insert_data_names = sorted(ProjectDataMetadata.__fields__.keys())
        insert_data_overrides = {
            "dataset_hash": literal(new_local_dataset_hash),
            "project_hash": literal(new_local_project_hash),
            "label_hash": func.gen_random_uuid() if engine.dialect.name == "postgresql" else None,
        }
        insert_data_unit_names = sorted(ProjectDataUnitMetadata.__fields__.keys())
        insert_data_unit_overrides = {"project_hash": literal(new_local_project_hash)}
        sess.execute(
            insert(ProjectDataMetadata).from_select(
                insert_data_names,
                select(
                    *[insert_data_overrides.get(k, getattr(ProjectDataMetadata, k)) for k in insert_data_names]
                ).where(in_op(ProjectDataMetadata.data_hash, subset_data_hashes)),
                include_defaults=False,
            )
        )
        sess.execute(
            insert(ProjectDataUnitMetadata).from_select(
                insert_data_unit_names,
                select(
                    *[
                        insert_data_unit_overrides.get(k, getattr(ProjectDataUnitMetadata, k))
                        for k in insert_data_unit_names
                    ]
                ).where(in_op(ProjectDataUnitMetadata.data_hash, subset_data_hashes)),
                include_defaults=False,
            )
        )
        sess.bulk_save_objects(new_collab)
        sess.add_all(new_data)
        sess.add_all(new_annotate)
        sess.add_all(new_data_extra)
        sess.add_all(new_annotate_extra)
        sess.commit()

        # FIXME: quick-abort
        return

        if project.project_remote_ssh_key_path is None:
            # Run for local project
            new_project_hash = uuid.uuid4()
            new_project_dataset_hash = uuid.uuid4()
            du_hash_to_label_hash_map = {
                data_hash: uuid.uuid4() for data_hash, du_hash, label_hash, dataset_hash in hashes
            }
            label_row_json_map = {}
        else:
            # Run for remote project
            original_project = get_encord_project(project.project_remote_ssh_key_path, str(project.project_hash))
            dataset_hash_map: Dict[uuid.UUID, Set[uuid.UUID]] = {}
            for data_hash, du_hash, label_hash, dataset_hash in hashes:
                dataset_hash_map.setdefault(dataset_hash, set()).add(data_hash)

            # Perform clone operation.
            new_project_hash_str: str = original_project.copy_project(
                new_title=item.project_name,
                new_description=item.project_description,
                copy_collaborators=True,
                copy_datasets=CopyDatasetOptions(
                    action=CopyDatasetAction.CLONE,
                    dataset_title=item.dataset_name,
                    dataset_description=item.dataset_description,
                    datasets_to_data_hashes_map={k: list(v) for k, v in dataset_hash_map.items()},
                ),
                copy_labels=CopyLabelsOptions(
                    accepted_label_statuses=[state for state in ReviewApprovalState],
                    accepted_label_hashes=list({label_hash for data_hash, du_hash, label_hash, dataset_hash in hashes}),
                ),
            )
            new_project = get_encord_project(project.project_remote_ssh_key_path, new_project_hash_str)
            new_project_hash = uuid.UUID(new_project_hash_str)
            new_project_label_rows = new_project.list_label_rows_v2()
            du_hash_to_label_hash_map = {
                label_row.data_hash: label_row.label_hash for label_row in new_project_label_rows
            }
            label_row_json_map = {
                label_row.data_hash: label_row.to_encord_dict() for label_row in new_project_label_rows
            }
            new_project_dataset_hashes = {label_row.dataset_hash for label_row in new_project_label_rows}
            if len(new_project_dataset_hashes) > 0:
                raise ValueError("Found multiple dataset hashes!")
            new_project_dataset_hash = uuid.UUID(list(new_project_dataset_hashes)[0])

        #
        # Populate all database tables
        #
        sess.add(
            Project(
                project_hash=new_project_hash,
                project_name=item.project_name,
                project_description=item.project_description,
                project_remote_ssh_key_path=project.project_remote_ssh_key_path,
                project_ontology=project.project_ontology,
            )
        )
        all_data = sess.exec(
            select(ProjectDataMetadata).where(
                ProjectDataMetadata.project_hash == project_hash, in_op(ProjectDataMetadata.data_hash, data_hashes)
            )
        )
        sess.add_all(
            [
                ProjectDataMetadata(
                    project_hash=new_project_hash,
                    data_hash=data_meta.data_hash,
                    label_hash=du_hash_to_label_hash_map[data_meta.data_hash],
                    dataset_hash=new_project_dataset_hash,
                    num_frames=data_meta.num_frames,
                    frames_per_second=data_meta.frames_per_second,
                    dataset_title=item.dataset_name,
                    data_type=data_meta.data_type,
                    label_row_json=label_row_json_map.get(data_meta.data_hash, data_meta.label_row_json),
                )
            ]
            for data_meta in all_data
        )

        sess.execute(
            insert(ProjectDataUnitMetadata)
            .from_select(
                list(set(ProjectDataUnitMetadata.__fields__.keys()) - {"project_hash"}),
                select(ProjectDataUnitMetadata).where(
                    ProjectDataUnitMetadata.project_hash == project_hash,
                    in_op(ProjectDataUnitMetadata.du_hash, du_hashes),
                ),
            )
            .values(
                project_hash=new_project_hash,
            )
        )

        # Commit changes
        sess.commit()


"""
debug = insert(ProjectDataUnitMetadata).from_select(
    list(ProjectDataUnitMetadata.__fields__.keys()),
    select(
        **{
            k: getattr(ProjectDataUnitMetadata, k)
            for k in ProjectDataUnitMetadata.__fields__.keys()
            if k != "project_hash"
        },
        project_hash=uuid.uuid4()
    ).where(
        ProjectDataUnitMetadata.project_hash == uuid.uuid4(),
        in_op(ProjectDataUnitMetadata.du_hash, [uuid.uuid4(), uuid.uuid4()])
    )
)
print(f"debug: {debug}")
"""


class UploadProjectToEncordPostAction(BaseModel):
    project_title: str
    project_description: Optional[str]
    dataset_title: str
    dataset_description: Optional[str]
    ontology_title: str
    ontology_description: Optional[str]


@router.post("/upload_to_encord")
def route_action_upload_project_to_encord(
    project_hash: uuid.UUID, item: UploadProjectToEncordPostAction, engine: Engine = Depends(dep_engine)
) -> None:
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError("Unknown project")
        if project.project_remote_ssh_key_path is not None:
            raise ValueError("Project already is bound to a remote")

        # Select all hashes present in the project
        hashes_query = select(
            ProjectDataUnitMetadata.data_hash,
            ProjectDataUnitMetadata.du_hash,
            ProjectDataMetadata.label_hash,
            ProjectDataMetadata.dataset_hash,
        ).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
            ProjectDataUnitMetadata.data_hash == ProjectDataMetadata.data_hash,
        )
        hashes = sess.exec(hashes_query).fetchall()
        du_hashes = [du_hash for data_hash, du_hash, label_hash, dataset_hash in hashes]

        # Create new encord-project

        # Create new dataset.

        # Create new encord ontology

        # Upload all data

        # Insert new
