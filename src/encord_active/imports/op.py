import uuid
from pathlib import Path
from typing import Optional

from encord import EncordUserClient
from encord.http.constants import RequestsSettings
from sqlmodel import Session, select

from ..db.models import Project, ProjectImportMetadata, get_engine
from .prediction.coco import import_coco_result
from .prediction.op import import_prediction
from .project.coco import import_coco
from .project.encord import import_encord
from .project.op import import_project, refresh_project


def import_coco_project(
    database_dir: Path,
    annotations_file_path: Path,
    images_dir_path: Optional[Path],
    store_symlinks: bool,
    store_data_locally: bool,
) -> None:
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    project_spec = import_coco(database_dir, annotations_file_path, images_dir_path, store_data_locally, store_symlinks)
    import_project(engine, database_dir, project_spec, "")


def import_encord_project(
    database_dir: Path, encord_project_hash: uuid.UUID, ssh_key: str, store_data_locally: bool
) -> None:
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    encord_client = EncordUserClient.create_with_ssh_private_key(
        ssh_key,
        requests_settings=RequestsSettings(max_retries=5),
    )
    encord_project = encord_client.get_project(str(encord_project_hash))
    project_spec = import_encord(encord_project, database_dir, store_data_locally)
    import_project(engine, database_dir, project_spec, ssh_key)


def refresh_encord_project(
    database_dir: Path,
    ssh_key: str,
    encord_project_hash: uuid.UUID,
    include_unlabeled: bool = False,
    force: bool = False,
) -> bool:
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        remote = sess.exec(select(Project.remote).where(Project.project_hash == encord_project_hash)).first()
        if not remote:
            raise ValueError(f"{encord_project_hash} does not correspond to a valid encord project")
    encord_client = EncordUserClient.create_with_ssh_private_key(
        ssh_key,
        requests_settings=RequestsSettings(max_retries=5),
    )
    encord_project = encord_client.get_project(str(encord_project_hash))
    project_spec = import_encord(
        encord_project, database_dir, include_unlabeled=include_unlabeled, store_data_locally=False
    )
    return refresh_project(engine, database_dir, ssh_key, project_spec, force=force)


def import_coco_prediction(
    database_dir: Path,
    predictions_file_path: Path,
    ssh_key: str,
    project_hash: uuid.UUID,
    prediction_name: str,
) -> None:
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        ontology = sess.exec(select(Project.ontology).where(Project.project_hash == project_hash)).first()
        if ontology is None:
            raise RuntimeError(f"Project hash: {project_hash} is missing from the database")
        metadata = sess.exec(
            select(ProjectImportMetadata.import_metadata).where(
                ProjectImportMetadata.project_hash == project_hash, ProjectImportMetadata.import_metadata_type == "COCO"
            )
        ).first()
        if metadata is None:
            raise RuntimeError("Project missing coco metadata to support importing coco prediction")
    coco_prediction = import_coco_result(
        ontology=dict(ontology),
        prediction_name=prediction_name,
        project_hash=project_hash,
        prediction_file=predictions_file_path,
        import_metadata=dict(metadata),
    )
    import_prediction(engine=engine, database_dir=database_dir, ssh_key=ssh_key, prediction=coco_prediction)


def import_encord_pickle_prediction(
    database_dir: Path,
    pickle_file_path: Path,
) -> None:
    raise ValueError()
