import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import encord
from encord import Dataset, EncordUserClient
from encord.constants.enums import DataType
from encord.http.constants import RequestsSettings
from encord.objects import OntologyStructure
from encord.orm.dataset import StorageLocation
from encord.orm.project import (
    CopyDatasetAction,
    CopyDatasetOptions,
    CopyLabelsOptions,
    ReviewApprovalState,
)
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import JSON, TEXT, bindparam, func, literal, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.engine import Engine
from sqlalchemy.sql.operators import in_op
from sqlmodel import Session, insert, select
from sqlmodel.sql.sqltypes import GUID
from tqdm import tqdm

from encord_active.analysis.config import create_analysis, default_torch_device
from encord_active.analysis.executor import SimpleExecutor
from encord_active.db.local_data import (
    db_uri_to_local_file_path,
    download_remote_to_file,
)
from encord_active.db.models import (
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
    ProjectEmbeddingReduction,
)
from encord_active.server.dependencies import dep_database_dir, dep_engine, dep_ssh_key
from encord_active.server.routers.queries import search_query
from encord_active.server.routers.queries.domain_query import TABLES_DATA
from encord_active.server.routers.queries.search_query import SearchFilters

router = APIRouter(
    prefix="/{project_hash}/actions",
)


SubsetTableType = TypeVar(
    "SubsetTableType",
    ProjectDataAnalytics,
    ProjectDataAnalyticsExtra,
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsExtra,
)


def _remote_project_clone(
    sess: Session,
    project_hash: uuid.UUID,
    subset_data_hashes: List[uuid.UUID],
    new_project: Project,
    ssh_key: str,
    dataset_title: str,
    dataset_description: Optional[str],
) -> Tuple[uuid.UUID, Dict[uuid.UUID, uuid.UUID]]:
    hash_lookup = sess.exec(
        select(ProjectDataMetadata.data_hash, ProjectDataMetadata.label_hash, ProjectDataMetadata.dataset_hash).where(
            ProjectDataMetadata.project_hash == project_hash,
            in_op(ProjectDataMetadata.data_hash, subset_data_hashes),
        )
    ).fetchall()
    dataset_hash_map: Dict[uuid.UUID, List[uuid.UUID]] = {}
    for data_hash, _label_hash, dataset_hash in hash_lookup:
        dataset_hash_map.setdefault(dataset_hash, []).append(data_hash)
    encord_client = encord.EncordUserClient.create_with_ssh_private_key(
        ssh_key,
        requests_settings=RequestsSettings(max_retries=5),
    )
    original_project = encord_client.get_project(str(project_hash))
    cloned_project_hash_str: str = original_project.copy_project(
        new_title=new_project.name,
        new_description=new_project.description,
        copy_collaborators=True,
        copy_datasets=CopyDatasetOptions(
            action=CopyDatasetAction.CLONE,
            dataset_title=dataset_title,
            dataset_description=dataset_description,
            datasets_to_data_hashes_map={str(k): [str(ve) for ve in v] for k, v in dataset_hash_map.items()},
        ),
        copy_labels=CopyLabelsOptions(
            accepted_label_statuses=[ReviewApprovalState(state) for state in ReviewApprovalState],
            accepted_label_hashes=[str(label_hash) for data_hash, label_hash, dataset_hash in hash_lookup],
        ),
    )
    cloned_project = encord_client.get_project(cloned_project_hash_str)

    # Change is applied at the end to keep label hash in sync.
    cloned_project_label_rows = cloned_project.list_label_rows_v2()
    for i in tqdm(range(0, len(cloned_project_label_rows), 50), desc="Fetching cloned label rows"):
        cloned_project_label_row_slice = cloned_project_label_rows[i : i + 50]
        init_bundle = cloned_project.create_bundle()
        for label_row in cloned_project_label_row_slice:
            label_row.initialise_labels(
                bundle=init_bundle, include_classification_feature_hashes=set(), include_object_feature_hashes=set()
            )
        init_bundle.execute()
    remap_label_hashes = {uuid.UUID(val.data_hash): uuid.UUID(val.label_hash) for val in cloned_project_label_rows}
    return uuid.UUID(cloned_project_hash_str), remap_label_hashes


def _update_label_hashes(
    sess: Session, project_hash: uuid.UUID, remap_data_hash_to_label_hash: Dict[uuid.UUID, uuid.UUID]
) -> None:
    if sess.bind is not None and sess.bind.dialect.name == "sqlite":
        # Doesn't support arrays, use json trick
        json_bind = JSON().bind_processor(dialect=sess.bind.dialect)
        sess.execute(
            text(
                "UPDATE project_data SET label_hash = un.value "
                "FROM json_each(:data_hash_mapping) as un "
                "WHERE project_data.data_hash = un.key "
                "AND project_data.project_hash = :project_hash"
            ).bindparams(bindparam("project_hash", GUID), bindparam("data_hash_mapping", JSON)),
            {
                "data_hash_mapping": json_bind(
                    {str(k).replace("-", ""): str(v).replace("-", "") for k, v in remap_data_hash_to_label_hash.items()}
                ),
                "project_hash": str(project_hash).replace("-", ""),
            },
        )
    else:
        from_data_hash_list = sorted(remap_data_hash_to_label_hash.keys())
        sess.execute(
            text(
                "UPDATE project_data SET label_hash = un.value "
                "FROM (SELECT unnest(:from_data_hash) AS key, unnest(:to_label_hash) AS value) as un "
                "WHERE project_data.data_hash = un.key "
                "AND project_data.project_hash = :project_hash"
            ).bindparams(
                bindparam("project_hash", GUID),
                bindparam("from_data_hash", ARRAY(GUID, dimensions=1)),
                bindparam("to_label_hash", ARRAY(GUID, dimensions=1)),
            ),
            {
                "from_data_hash": list(from_data_hash_list),
                "to_label_hash": [remap_data_hash_to_label_hash[k] for k in from_data_hash_list],
                "project_hash": project_hash,
            },
        )


class CreateProjectSubsetPostAction(BaseModel):
    project_title: str
    project_description: Optional[str]
    dataset_title: str
    dataset_description: Optional[str]
    filters: SearchFilters


@router.post("/create_project_subset")
def route_action_create_project_subset(
    project_hash: uuid.UUID,
    item: CreateProjectSubsetPostAction,
    engine: Engine = Depends(dep_engine),
    database_dir: Path = Depends(dep_database_dir),
    ssh_key: str = Depends(dep_ssh_key),
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

        remap_label_hashes: Dict[uuid.UUID, uuid.UUID] = {}
        if current_project.remote:
            new_local_project_hash, remap_label_hashes = _remote_project_clone(
                sess=sess,
                project_hash=project_hash,
                subset_data_hashes=subset_data_hashes,
                new_project=new_project,
                ssh_key=ssh_key,
                dataset_title=item.dataset_title,
                dataset_description=item.dataset_description,
            )
            new_project.project_hash = new_local_project_hash

        # Use this info to generate NEW values to insert
        metric_engine = SimpleExecutor(create_analysis(default_torch_device()))
        new_analytics = metric_engine.execute_from_db_subset(
            database_dir=database_dir,
            project_hash=new_local_project_hash,
            project_ontology=current_project.ontology,
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

        guid_bind = GUID().bind_processor(dialect=engine.dialect)
        overrides = {"project_hash": literal(guid_bind(new_local_project_hash))}
        insert_data_overrides = {
            **overrides,
            "dataset_hash": literal(guid_bind(new_local_dataset_hash)),
            "label_hash": func.gen_random_uuid()
            if engine.dialect.name == "postgresql"
            else func.lower(func.hex(func.randomblob(16))),
        }
        insert_data_unit_names = sorted(ProjectDataUnitMetadata.__fields__.keys())
        sess.execute(
            insert(ProjectDataMetadata).from_select(
                insert_data_names,
                select(  # type: ignore
                    *[insert_data_overrides.get(k, getattr(ProjectDataMetadata, k)) for k in insert_data_names]
                ).where(
                    in_op(ProjectDataMetadata.data_hash, subset_data_hashes),
                    ProjectDataMetadata.project_hash == project_hash,
                ),
                include_defaults=False,
            )
        )
        sess.execute(
            insert(ProjectDataUnitMetadata).from_select(
                insert_data_unit_names,
                select(  # type: ignore
                    *[overrides.get(k, getattr(ProjectDataUnitMetadata, k)) for k in insert_data_unit_names]
                ).where(
                    in_op(ProjectDataUnitMetadata.data_hash, subset_data_hashes),
                    ProjectDataUnitMetadata.project_hash == project_hash,
                ),
                include_defaults=False,
            )
        )
        sess.bulk_save_objects(new_collab)
        sess.bulk_save_objects(new_data)
        sess.bulk_save_objects(new_annotate)
        sess.bulk_save_objects(new_data_extra)
        sess.bulk_save_objects(new_annotate_extra)

        insert_reduction_names = sorted(ProjectEmbeddingReduction.__fields__.keys())
        insert_reduced_data_names = sorted(ProjectDataAnalyticsReduced.__fields__.keys())
        insert_reduced_annotation_names = sorted(ProjectAnnotationAnalyticsReduced.__fields__.keys())
        insert_reduction_overrides = {
            **overrides,
            "reduction_hash": literal(guid_bind(uuid.uuid4())),
        }
        sess.execute(
            insert(ProjectEmbeddingReduction).from_select(
                insert_reduction_names,
                select(  # type: ignore
                    *[
                        insert_reduction_overrides.get(k, getattr(ProjectEmbeddingReduction, k))
                        for k in insert_reduction_names
                    ]
                ).where(
                    ProjectEmbeddingReduction.project_hash == project_hash,
                ),
                include_defaults=False,
            )
        )
        sess.execute(
            insert(ProjectDataAnalyticsReduced).from_select(
                insert_reduced_data_names,
                select(  # type: ignore
                    *[
                        insert_reduction_overrides.get(k, getattr(ProjectDataAnalyticsReduced, k))
                        for k in insert_reduced_data_names
                    ]
                ).where(
                    in_op(ProjectDataAnalyticsReduced.du_hash, subset_data_hashes),
                    ProjectDataAnalyticsReduced.project_hash == project_hash,
                ),
                include_defaults=False,
            )
        )
        sess.execute(
            insert(ProjectAnnotationAnalyticsReduced).from_select(
                insert_reduced_annotation_names,
                select(  # type: ignore
                    *[
                        insert_reduction_overrides.get(k, getattr(ProjectAnnotationAnalyticsReduced, k))
                        for k in insert_reduced_annotation_names
                    ]
                ).where(
                    in_op(ProjectAnnotationAnalyticsReduced.du_hash, subset_data_hashes),
                    ProjectAnnotationAnalyticsReduced.project_hash == project_hash,
                ),
                include_defaults=False,
            )
        )

        if len(remap_label_hashes) > 0:
            _update_label_hashes(sess, project_hash, remap_label_hashes)

        sess.commit()


def _file_path_for_upload(
    tempdir: Path,
    data_uri: Optional[str],
    database_dir: Path,
) -> Path:
    if data_uri is None:
        raise RuntimeError("Attempting to upload remote file to encord, this is not supported")
    opt_path = db_uri_to_local_file_path(data_uri, database_dir)
    if opt_path is not None:
        return opt_path
    else:
        temp_path = tempdir / str(uuid.uuid4())
        download_remote_to_file(data_uri, temp_path)
        return temp_path


def _upload_data_to_encord(
    dataset: Dataset,
    tempdir: Path,
    database_dir: Path,
    data_hash: uuid.UUID,
    du_meta_list: List["_InternalDuUploadState"],
    data_hash_data_type_map: Dict[uuid.UUID, DataType],
) -> None:
    # FIXME: add sanity check assertions for certain upload actions (rule out impossible state)
    du_meta_list.sort(key=lambda d: (d.data_hash, d.du_hash, d.frame))
    du_entry0 = du_meta_list[0]
    if du_entry0.data_type == "image" and len(du_meta_list) == 1:
        data_hash_data_type_map[data_hash] = DataType.IMAGE
        dataset.upload_image(
            _file_path_for_upload(tempdir=tempdir, data_uri=du_entry0.data_uri, database_dir=database_dir),
            title=str(data_hash),
        )
    elif du_entry0.data_type in {"image", "image_group"}:
        data_hash_data_type_map[data_hash] = DataType.IMG_GROUP
        dataset.create_image_group(
            file_paths=[
                _file_path_for_upload(
                    tempdir=tempdir, data_uri=du_entry_image.data_uri, database_dir=database_dir
                ).as_posix()
                for du_entry_image in du_meta_list
            ],
            title=str(data_hash),
            create_video=False,
        )
    elif du_entry0.data_type == "video" and du_entry0.data_uri_is_video:
        data_hash_data_type_map[data_hash] = DataType.VIDEO
        dataset.upload_video(
            _file_path_for_upload(tempdir=tempdir, data_uri=du_entry0.data_uri, database_dir=database_dir).as_posix(),
            title=str(data_hash),
        )
    else:
        raise RuntimeError(f"Unsupported config for upload to encord: {du_entry0.data_type}: {len(du_meta_list)}")


def _build_label_row_json(
    data: ProjectDataMetadata,
    du_grouped_list: Dict[uuid.UUID, List[ProjectDataUnitMetadata]],
    label_row: encord.objects.LabelRowV2,
    uploaded_dataset_api: encord.Dataset,
    du_hash_to_data_link_map: Dict[uuid.UUID, Optional[str]],
    du_hash_map: Dict[uuid.UUID, uuid.UUID],
    data_hash_data_type_map: Dict[uuid.UUID, DataType],
) -> dict:
    return {
        "label_hash": label_row.label_hash,
        "dataset_hash": str(uploaded_dataset_api.dataset_hash),
        "dataset_title": str(uploaded_dataset_api.title),
        "data_title": str(label_row.data_title),
        "data_type": str(data_hash_data_type_map[data.data_hash].value),
        "data_hash": str(label_row.data_hash),
        "label_status": "LABEL_IN_PROGRESS",
        "created_at": str(data.created_at),
        "last_edited_at": str(data.last_edited_at),
        "object_answers": data.object_answers,
        "classification_answers": data.classification_answers,
        "object_actions": {},
        "data_units": {
            str(du_hash_map[du_hash]): {
                "data_hash": str(du_hash_map[du_hash]),
                "data_sequence": int(data_sequence),
                "data_title": str(du_group_list[0].data_title),
                "data_type": str(du_group_list[0].data_type),
                "data_link": str(du_hash_to_data_link_map[du_hash_map[du_hash]] or ""),
                "width": int(du_group_list[0].width),
                "height": int(du_group_list[0].height),
                "labels": {
                    "objects": du_group_list[0].objects,
                    "classifications": du_group_list[0].classifications,
                },
            }
            if du_group_list[0].data_type != "video" and len(du_group_list) == 1
            else {
                "data_hash": str(du_hash_map[du_hash]),
                "width": int(du_group_list[0].width),
                "height": int(du_group_list[0].height),
                "labels": {
                    str(du_group_entry.frame): {
                        "objects": du_group_entry.objects,
                        "classifications": du_group_entry.classifications,
                    }
                    for du_group_entry in du_group_list
                },
            }
            for data_sequence, (du_hash, du_group_list) in enumerate(du_grouped_list.items())
        },
    }


class _InternalDuUploadState(BaseModel):
    data_hash: uuid.UUID
    du_hash: uuid.UUID
    frame: int
    data_uri: Optional[str]
    data_uri_is_video: bool
    label_hash: uuid.UUID
    dataset_hash: uuid.UUID
    data_type: str
    data_title: str


def _upload_dataset_data_to_encord(
    hashes: List[_InternalDuUploadState],
    uploaded_dataset_api: encord.Dataset,
    database_dir: Path,
    data_hash_map: Dict[uuid.UUID, uuid.UUID],
    du_hash_map: Dict[uuid.UUID, uuid.UUID],
    data_hash_data_type_map: Dict[uuid.UUID, DataType],
) -> Dict[uuid.UUID, Optional[str]]:
    # Upload all data to the dataset.
    data_upload_state: Dict[uuid.UUID, List[_InternalDuUploadState]] = {}
    du_hash_to_data_link_map = {}
    for data_hash_entry in hashes:
        if data_hash_entry.data_uri is None:
            raise ValueError(f"{data_hash_entry.du_hash} / {data_hash_entry.frame} has null data_uri")
        data_upload_state.setdefault(data_hash_entry.data_hash, []).append(data_hash_entry)

    with tempfile.TemporaryDirectory() as tempdir:
        for data_hash, upload_du_list in tqdm(data_upload_state.items(), desc="Uploading to encord"):
            _upload_data_to_encord(
                uploaded_dataset_api, Path(tempdir), database_dir, data_hash, upload_du_list, data_hash_data_type_map
            )

    # The data titles can now be used to generate a data hash map.
    uploaded_dataset_rows = uploaded_dataset_api.list_data_rows()
    for uploaded_ds_row in uploaded_dataset_rows:
        ds_old_data_hash = uuid.UUID(uploaded_ds_row.title)
        ds_new_data_hash = uuid.UUID(uploaded_ds_row.uid)
        upload_state = data_upload_state[ds_old_data_hash]

        data_hash_map[ds_old_data_hash] = ds_new_data_hash
        uploaded_ds_images = uploaded_ds_row.images_data
        if uploaded_ds_images is not None:
            for iter_image, iter_upload_state in zip(uploaded_ds_images, upload_state):
                iter_image.title = iter_upload_state.data_title
                du_hash_to_data_link_map[uuid.UUID(iter_image.data_hash)] = iter_image.file_link
            uploaded_ds_row.title = f"Image Group[{upload_state[0].data_title}]"
        else:
            du_hash_map[ds_old_data_hash] = ds_new_data_hash
            uploaded_ds_row.title = upload_state[0].data_title
            du_hash_to_data_link_map[uuid.UUID(uploaded_ds_row.uid)] = uploaded_ds_row.file_link
        uploaded_ds_row.save()

    return du_hash_to_data_link_map


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
            ProjectDataUnitMetadata.data_title,
        ).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
            ProjectDataUnitMetadata.data_hash == ProjectDataMetadata.data_hash,
        )
        hashes_raw: List[
            Tuple[uuid.UUID, uuid.UUID, int, Optional[str], bool, uuid.UUID, uuid.UUID, str, str]
        ] = sess.exec(hashes_query).fetchall()
        hashes: List[_InternalDuUploadState] = [
            _InternalDuUploadState(
                data_hash=data_hash,
                du_hash=du_hash,
                frame=frame,
                data_uri=data_uri,
                data_uri_is_video=data_uri_is_video,
                label_hash=label_hash,
                dataset_hash=dataset_hash,
                data_type=data_type,
                data_title=data_title,
            )
            for (
                data_hash,
                du_hash,
                frame,
                data_uri,
                data_uri_is_video,
                label_hash,
                dataset_hash,
                data_type,
                data_title,
            ) in hashes_raw
        ]

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
    data_hash_map: Dict[uuid.UUID, uuid.UUID] = {}
    du_hash_map: Dict[uuid.UUID, uuid.UUID] = {}
    data_hash_data_type_map: Dict[uuid.UUID, DataType] = {}
    du_hash_to_data_link_map: Dict[uuid.UUID, Optional[str]] = _upload_dataset_data_to_encord(
        hashes=hashes,
        uploaded_dataset_api=uploaded_dataset_api,
        database_dir=database_dir,
        data_hash_map=data_hash_map,
        du_hash_map=du_hash_map,
        data_hash_data_type_map=data_hash_data_type_map,
    )

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

    # Upload all labels to encord
    label_upload_state: Dict[uuid.UUID, Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]] = {
        data.data_hash: (data, []) for data in data_results
    }
    for du in du_results:
        data, du_list = label_upload_state[du.data_hash]
        du_list.append(du)

    data_hash_map_rev = {v: k for k, v in data_hash_map.items()}
    data_hash_to_label_hash_map: Dict[uuid.UUID, uuid.UUID] = {}
    uploaded_project_api = encord_client.get_project(uploaded_project_hash)
    all_label_rows = uploaded_project_api.list_label_rows_v2()
    for i in tqdm(range(0, len(all_label_rows), 50), desc="Uploading labels"):
        slice_label_rows = all_label_rows[i : i + 50]
        # Initialize bundle
        init_bundle = uploaded_project_api.create_bundle()
        for label_row in slice_label_rows:
            label_row.initialise_labels(bundle=init_bundle)
        init_bundle.execute()
        # Save state
        bundle = uploaded_project_api.create_bundle()
        for label_row in slice_label_rows:
            data_hash = uuid.UUID(label_row.data_hash)
            data, du_list = label_upload_state[data_hash_map_rev[data_hash]]
            du_grouped_list: Dict[uuid.UUID, List[ProjectDataUnitMetadata]] = {}
            for du in du_list:
                du_grouped_list.setdefault(du.du_hash, []).append(du)
            du_group_list: List[ProjectDataUnitMetadata]
            label_row_json = _build_label_row_json(
                data=data,
                du_grouped_list=du_grouped_list,
                label_row=label_row,
                uploaded_dataset_api=uploaded_dataset_api,
                du_hash_to_data_link_map=du_hash_to_data_link_map,
                du_hash_map=du_hash_map,
                data_hash_data_type_map=data_hash_data_type_map,
            )
            label_row.from_labels_dict(label_row_json)
            label_row.save(bundle=bundle)
        bundle.execute()
        # Populate label_hash map once labels are created.
        for label_row in slice_label_rows:
            data_hash_to_label_hash_map[uuid.UUID(label_row.data_hash)] = uuid.UUID(label_row.label_hash)

    # Now the database contains incorrect hashes (project_hash, data_hash, du_hash, label_hash, dataset_hash).
    # Update the database to commit all these changes.
    with Session(engine) as sess:
        sess.execute(
            text(
                "UPDATE project SET project_hash = :new_project_hash, remote = TRUE, "
                "name = :new_name, description = :new_description "
                "WHERE project_hash = :old_project_hash"
            ).bindparams(
                bindparam("new_project_hash", GUID),
                bindparam("old_project_hash", GUID),
                bindparam("new_name", TEXT),
                bindparam("new_description", TEXT),
            ),
            {
                "new_project_hash": uuid.UUID(uploaded_project_hash),
                "old_project_hash": project_hash,
                "new_name": item.project_title,
                "new_description": item.project_description or "",
            },
        )
        sess.execute(
            text(
                "UPDATE project_data SET dataset_hash = :new_project_dataset "
                "WHERE project_data.project_hash = :project_hash"
            ).bindparams(bindparam("project_hash", GUID), bindparam("new_project_dataset", GUID)),
            {
                "new_project_dataset": uuid.UUID(uploaded_dataset.dataset_hash),
                "project_hash": uuid.UUID(uploaded_project_hash),
            },
        )
        if engine.dialect.name == "sqlite":
            sess.execute(
                text(
                    "UPDATE project_data SET data_hash = un.value "
                    "FROM json_each(:data_hash_mapping) as un "
                    "WHERE project_data.data_hash = un.key "
                    "AND project_data.project_hash = :project_hash"
                ).bindparams(bindparam("project_hash", GUID), bindparam("data_hash_mapping", JSON)),
                {
                    "data_hash_mapping": {
                        str(k).replace("-", ""): str(v).replace("-", "") for k, v in data_hash_map.items()
                    },
                    "project_hash": uuid.UUID(uploaded_project_hash),
                },
            )
            sess.execute(
                text(
                    "UPDATE project_data_units SET du_hash = un.value "
                    "FROM json_each(:du_hash_mapping) as un "
                    "WHERE project_data_units.du_hash = un.key "
                    "AND project_data_units.project_hash = :project_hash"
                ).bindparams(bindparam("project_hash", GUID), bindparam("du_hash_mapping", JSON)),
                {
                    "du_hash_mapping": {
                        str(k).replace("-", ""): str(v).replace("-", "") for k, v in du_hash_map.items()
                    },
                    "project_hash": uuid.UUID(uploaded_project_hash),
                },
            )
        else:
            from_data_hash_list = sorted(data_hash_map.keys())
            sess.execute(
                text(
                    "UPDATE project_data SET data_hash = un.value "
                    "FROM (SELECT unnest(:from_data_hash) AS key, unnest(:to_data_hash) AS value) as un "
                    "WHERE project_data.data_hash = un.key "
                    "AND project_data.project_hash = :project_hash",
                ).bindparams(
                    bindparam("project_hash", GUID),
                    bindparam("from_data_hash", ARRAY(GUID, dimensions=1)),
                    bindparam("to_data_hash", ARRAY(GUID, dimensions=1)),
                ),
                {
                    "from_data_hash": from_data_hash_list,
                    "to_data_hash": [data_hash_map[k] for k in from_data_hash_list],
                    "project_hash": uuid.UUID(uploaded_project_hash),
                },
            )
            from_du_hash_list = sorted(du_hash_map.keys())
            sess.execute(
                text(
                    "UPDATE project_data_units SET du_hash = un.value "
                    "FROM (SELECT unnest(:from_du_hash) AS key, unnest(:to_du_hash) AS value) as un "
                    "WHERE project_data_units.du_hash = un.key "
                    "AND project_data_units.project_hash = :project_hash",
                ).bindparams(
                    bindparam("project_hash", GUID),
                    bindparam("from_du_hash", ARRAY(GUID, dimensions=1)),
                    bindparam("to_du_hash", ARRAY(GUID, dimensions=1)),
                ),
                {
                    "from_du_hash": from_du_hash_list,
                    "to_du_hash": [du_hash_map[k] for k in from_du_hash_list],
                    "project_hash": uuid.UUID(uploaded_project_hash),
                },
            )
        _update_label_hashes(sess, uuid.UUID(uploaded_project_hash), data_hash_to_label_hash_map)
        sess.commit()
