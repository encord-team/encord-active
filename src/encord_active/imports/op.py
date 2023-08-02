import uuid
from pathlib import Path
from typing import Optional

from encord import EncordUserClient
from encord.http.constants import RequestsSettings

from ..db.models import get_engine
from .project.coco import import_coco
from .project.encord import import_encord
from .project.op import import_project


def import_coco_project(
    database_dir: Path,
    annotations_file_path: Path,
    images_dir_path: Optional[Path],
    store_symlinks: bool,  # FIXME: use argument!
    store_data_locally: bool,
) -> None:
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    project_spec = import_coco(database_dir, annotations_file_path, images_dir_path, store_data_locally, store_symlinks)
    import_project(engine, database_dir, project_spec)


def import_encord_project(
    database_dir: Path, encord_project_hash: uuid.UUID, ssh_key_path: Path, store_data_locally: bool
) -> None:
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    encord_client = EncordUserClient.create_with_ssh_private_key(
        Path(ssh_key_path).read_text(encoding="utf-8"),
        requests_settings=RequestsSettings(max_retries=5),
    )
    encord_project = encord_client.get_project(str(encord_project_hash))
    project_spec = import_encord(encord_project, ssh_key_path, database_dir, store_data_locally)
    import_project(engine, database_dir, project_spec)


def import_coco_prediction(
    database_dir: Path,
    predictions_file_path: Path,
) -> None:
    raise ValueError()


def import_encord_pickle_prediction(
    database_dir: Path,
    pickle_file_path: Path,
) -> None:
    raise ValueError()
