import uuid
from typing import Dict, List, Set

from encord.orm.project import (
    CopyDatasetAction,
    CopyDatasetOptions,
    CopyLabelsOptions,
    ReviewApprovalState,
)
from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy.sql.operators import in_op
from sqlmodel import Session, insert, select

from encord_active.db.models import (
    Project,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
)
from encord_active.lib.encord.utils import get_encord_project
from encord_active.server.routers.project2_engine import engine

router = APIRouter(
    prefix="/{project_hash}/actions",
)


class CreateProjectSubsetPostAction(BaseModel):
    project_name: str
    project_description: str
    dataset_name: str
    dataset_description: str
    du_hashes: List[str]


@router.post("/create_project_subset")
def create_active_subset(project_hash: uuid.UUID, item: CreateProjectSubsetPostAction):
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError("Unknown project")

        data_hashes = select(ProjectDataUnitMetadata.data_hash).where(
            ProjectDataUnitMetadata.project_hash == project_hash, in_op(ProjectDataUnitMetadata.du_hash, item.du_hashes)
        )
        # Map to data_hashes and back again to duplicate expected behaviour.
        hashes_query = select(
            ProjectDataUnitMetadata.data_hash,
            ProjectDataUnitMetadata.du_hash,
            ProjectDataMetadata.label_hash,
            ProjectDataMetadata.dataset_hash,
        ).where(
            ProjectDataUnitMetadata.project_hash == project_hash,
            ProjectDataUnitMetadata.data_hash == ProjectDataMetadata.data_hash,
            in_op(ProjectDataMetadata.data_hash, data_hashes),
        )
        hashes = sess.exec(hashes_query).fetchall()
        du_hashes = [du_hash for data_hash, du_hash, label_hash, dataset_hash in hashes]

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
    project_name: str
    project_description: str
    dataset_name: str
    dataset_description: str
    ontology_name: str
    ontology_description: str


@router.post("/upload_to_encord")
def upload_project_to_encord(project_hash: uuid.UUID, item: UploadProjectToEncordPostAction):
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
