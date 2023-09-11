import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import requests
from encord import Dataset, EncordUserClient
from encord.http.constants import RequestsSettings
from encord.objects import OntologyStructure
from encord.orm.dataset import StorageLocation
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, literal
from sqlalchemy.engine import Engine
from sqlalchemy.sql.operators import in_op
from sqlmodel import Session, insert, select
from tqdm import tqdm

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
)
from encord_active.lib.common.data_utils import url_to_file_path
from encord_active.server.dependencies import dep_database_dir, dep_engine, dep_ssh_key
from encord_active.server.routers.queries import search_query
from encord_active.server.routers.queries.domain_query import TABLES_DATA
from encord_active.server.routers.queries.search_query import SearchFilters

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
    ProjectDataAnalytics,
    ProjectDataAnalyticsExtra,
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsExtra,
)


@router.post("/create_project_subset")
def route_action_create_project_subset(
    project_hash: uuid.UUID,
    item: CreateProjectSubsetPostAction,
    engine: Engine = Depends(dep_engine),
    database_dir: Path = Depends(dep_database_dir),
) -> None:
    with Session(engine) as sess:
        current_project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if current_project is None:
            raise ValueError("Unknown project")
        new_local_project_hash = uuid.uuid4()
        new_project = Project(
            project_hash=new_local_project_hash,
            name=item.project_title,
            description=item.project_description or "",
            remote=current_project.remote,
            ontology=current_project.ontology,
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
        new_analytics = metric_engine.execute_from_db_subset(
            database_dir=database_dir,
            project_hash=new_local_project_hash,
            precalculated_data=subset_data_analytics,
            precalculated_data_extra=subset_data_analytics_extra,
            precalculated_annotation=subset_annotation_analytics,
            precalculated_annotation_extra=subset_annotation_analytics_extra,
            precalculated_collaborators=current_collaborators,
        )
        (
            new_data,
            new_data_extra,
            new_data_derived,
            new_annotate,
            new_annotate_extra,
            new_annotate_derived,
            new_collab,
        ) = new_analytics

        # Insert state into the database
        new_local_dataset_hash = uuid.uuid4()
        sess.execute(insert(Project).values(**{k: getattr(new_project, k) for k in Project.__fields__}))
        insert_data_names = sorted(ProjectDataMetadata.__fields__.keys())
        insert_data_overrides = {
            "dataset_hash": literal(new_local_dataset_hash),
            "project_hash": literal(new_local_project_hash),
            "label_hash": func.gen_random_uuid()
            if engine.dialect.name == "postgresql"
            else func.lower(func.hex(func.randomblob(16))),
        }
        insert_data_unit_names = sorted(ProjectDataUnitMetadata.__fields__.keys())
        insert_data_unit_overrides = {"project_hash": literal(new_local_project_hash)}
        sess.execute(
            insert(ProjectDataMetadata).from_select(
                insert_data_names,
                select(  # type: ignore
                    *[insert_data_overrides.get(k, getattr(ProjectDataMetadata, k)) for k in insert_data_names]
                ).where(in_op(ProjectDataMetadata.data_hash, subset_data_hashes)),
                include_defaults=False,
            )
        )
        sess.execute(
            insert(ProjectDataUnitMetadata).from_select(
                insert_data_unit_names,
                select(  # type: ignore
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
        """
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


def _file_path_for_upload(
    tempdir: Path,
    data_uri: str,
    database_dir: Path,
) -> Path:
    opt_path = url_to_file_path(data_uri, database_dir)
    if opt_path is not None:
        return opt_path
    else:
        temp_path = tempdir / str(uuid.uuid4())
        with open(temp_path, "xb") as file:
            r = requests.get(data_uri, stream=True)
            if r.status_code != 200:
                raise ConnectionError(f"Something happened, couldn't download file from: {data_uri}")
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
            file.flush()
        return temp_path


def _upload_data_to_encord(
    dataset: Dataset,
    tempdir: Path,
    database_dir: Path,
    data_hash: uuid.UUID,
    du_meta_list: List[Tuple[uuid.UUID, int, str, bool, str]],
) -> None:
    # FIXME: add sanity check assertions for certain upload actions (rule out impossible state)
    du_meta_list.sort(key=lambda x: x[1])
    du_hash0, frame0, data_uri0, data_uri_is_video0, data_type = du_meta_list[0]
    if data_type == "image" and len(du_meta_list) == 0:
        dataset.upload_image(
            _file_path_for_upload(tempdir=tempdir, data_uri=data_uri0, database_dir=database_dir), title=str(data_hash)
        )
    elif data_type in {"image", "image_group"}:
        dataset.create_image_group(
            file_paths=[
                _file_path_for_upload(tempdir=tempdir, data_uri=data_uriN, database_dir=database_dir).as_posix()
                for du_hashN, frameN, data_uriN, data_uri_is_videoN, data_typeN in du_meta_list
            ],
            title=str(data_hash),
            create_video=False,
        )
    elif data_type == "video" and data_uri_is_video0:
        dataset.upload_video(
            _file_path_for_upload(tempdir=tempdir, data_uri=data_uri0, database_dir=database_dir).as_posix(),
            title=str(data_hash),
        )
    else:
        raise RuntimeError("Unsupported config for upload to encord")


class UploadProjectToEncordPostAction(BaseModel):
    project_title: str
    project_description: Optional[str]
    dataset_title: str
    dataset_description: Optional[str]
    ontology_title: str
    ontology_description: Optional[str]


@router.post("/upload_to_encord")
def route_action_upload_project_to_encord(
    project_hash: uuid.UUID,
    item: UploadProjectToEncordPostAction,
    engine: Engine = Depends(dep_engine),
    ssh_key: str = Depends(dep_ssh_key),
    database_dir: Path = Depends(dep_database_dir),
) -> None:
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError("Unknown project")
        if project.remote:
            raise ValueError("Project already is bound to a remote")

        # Select all hashes present in the project
        hashes_query = select(  # type: ignore
            ProjectDataUnitMetadata.data_hash,
            ProjectDataUnitMetadata.du_hash,
            ProjectDataUnitMetadata.frame,
            ProjectDataUnitMetadata.data_uri,
            ProjectDataUnitMetadata.data_uri_is_video,
            ProjectDataMetadata.label_hash,
            ProjectDataMetadata.dataset_hash,
            ProjectDataMetadata.data_type,
        ).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
            ProjectDataUnitMetadata.data_hash == ProjectDataMetadata.data_hash,
        )
        hashes: List[Tuple[uuid.UUID, uuid.UUID, int, Optional[str], bool, uuid.UUID, uuid.UUID, str]] = sess.exec(
            hashes_query
        ).fetchall()

    # Create new encord-project
    encord_client = EncordUserClient.create_with_ssh_private_key(
        ssh_key,
        requests_settings=RequestsSettings(max_retries=5),
    )

    # Create ontology & dataset.
    uploaded_ontology = encord_client.create_ontology(
        title=item.ontology_title,
        description=item.ontology_description or "",
        structure=OntologyStructure.from_dict(project.ontology),
    )
    uploaded_dataset = encord_client.create_dataset(
        dataset_title=item.dataset_title,
        dataset_type=StorageLocation.CORD_STORAGE,
        dataset_description=item.dataset_description,
    )
    uploaded_dataset_api = encord_client.get_dataset(uploaded_dataset.dataset_hash)

    # Upload all data to the dataset.
    data_upload_state: Dict[uuid.UUID, List[Tuple[uuid.UUID, int, str, bool, str]]] = {}
    for data_hash, du_hash, frame, data_uri, data_uri_is_video, label_hash, dataset_hash, data_type in hashes:
        if data_uri is None:
            raise ValueError(f"{du_hash} / {frame} has null data_uri")
        data_upload_state.setdefault(data_hash, []).append((du_hash, frame, data_uri, data_uri_is_video, data_type))

    with tempfile.TemporaryDirectory() as tempdir:
        for data_hash, upload_du_list in tqdm(data_upload_state.items(), desc="Uploading to encord"):
            _upload_data_to_encord(uploaded_dataset_api, Path(tempdir), database_dir, data_hash, upload_du_list)

    # Now that all data has been uploaded, create the encord project.
    uploaded_project_hash = encord_client.create_project(
        project_title=item.project_title,
        dataset_hashes=[uploaded_dataset.dataset_hash],
        project_description=item.project_description or "",
        ontology_hash=uploaded_ontology.ontology_hash,
    )

    # Upload all materialized label rows to the project.
    with Session(engine) as sess:
        du_query = select(ProjectDataUnitMetadata).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
        )
        du_results = sess.exec(du_query).fetchall()
        data_query = select(ProjectDataMetadata).where(ProjectDataMetadata.project_hash == project_hash)
        data_results = sess.exec(data_query).fetchall()
        label_upload_state: Dict[uuid.UUID, Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]] = {
            data.data_hash: (data, []) for data in data_results
        }
        for du in du_results:
            data, du_list = label_upload_state[du.data_hash]
            du_list.append(du)

        uploaded_project_api = encord_client.get_project(uploaded_project_hash)
        all_label_rows = uploaded_project_api.list_label_rows_v2()
        for i in range(0, len(all_label_rows), 50):
            slice_label_rows = all_label_rows[i : i + 50]
            # bundle = uploaded_project_api.create_bundle()
            for label_row in slice_label_rows:
                data_hash = uuid.UUID(label_row.data_hash)
                data, du_list = label_upload_state[data_hash]

                # label_row.save(bundle=bundle)
            # bundle.execute()

    # Now the database contains incorrect hashes (project_hash, data_hash, du_hash, label_hash, dataset_hash).
    # Update the database to commit all these changes.
