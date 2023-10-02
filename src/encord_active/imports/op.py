import uuid
from pathlib import Path
from typing import List, Optional

from encord import EncordUserClient
from encord.http.constants import RequestsSettings
from sqlmodel import Session, select

from encord_active.db.models.project_prediction import ProjectPrediction
from encord_active.imports.prediction.legacy import import_legacy_predictions

from ..db.models import Project, ProjectImportMetadata, get_engine
from ..public.label_transformer import LabelTransformer
from .prediction.coco import import_coco_result
from .prediction.op import PredictionImportSpec, import_prediction
from .project.coco import import_coco
from .project.encord import import_encord
from .project.label_transformer import import_label_transformer
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
    import_project(engine, database_dir, project_spec, None)


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


def import_local_project(
    database_dir: Path,
    files: List[Path],
    project_name: str,
    symlinks: bool,
    label_transformer: Optional[LabelTransformer],
    label_paths: List[Path],
) -> None:
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    project_spec = import_label_transformer(
        database_dir=database_dir,
        files=files,
        project_name=project_name,
        symlinks=symlinks,
        label_transformer=label_transformer,
        label_paths=label_paths,
    )
    import_project(engine, database_dir, project_spec, None)


def refresh_encord_project(
    database_dir: Path,
    ssh_key: str,
    encord_project_hash: uuid.UUID,
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
    project_spec = import_encord(encord_project, database_dir, store_data_locally=False)
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


def import_legacy_prediction(
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
    coco_prediction = import_legacy_predictions(
        ontology=dict(ontology),
        prediction_name=prediction_name,
        project_hash=project_hash,
        prediction_file=predictions_file_path,
    )
    import_prediction(engine=engine, database_dir=database_dir, ssh_key=ssh_key, prediction=coco_prediction)
