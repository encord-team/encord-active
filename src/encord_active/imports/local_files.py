import shutil
import uuid
from pathlib import Path
from typing import Union

from encord_active.db.local_data import (
    download_remote_to_file,
    file_path_to_database_uri,
)


def get_local_file(database_dir: Path) -> Path:
    local_data_folder = database_dir / "local-data"
    local_data_folder.mkdir(exist_ok=True, parents=False)
    local_file = local_data_folder / str(uuid.uuid4())
    if local_file.exists():
        raise ValueError(f"uuid collision in local-data: {local_file.name}")
    return local_file


def download_to_local_file(database_dir: Path, local_file: Path, url: str) -> str:
    download_remote_to_file(url, local_file)
    return file_path_to_database_uri(local_file, database_dir)


def get_data_uri(
    url_or_path: Union[str, Path],
    store_data_locally: bool,
    store_symlinks: bool,
    database_dir: Path,
) -> str:
    local_file = get_local_file(database_dir)
    if isinstance(url_or_path, Path):
        if store_data_locally and store_symlinks:
            local_file.symlink_to(url_or_path)
            return file_path_to_database_uri(local_file, database_dir)
        elif store_data_locally:
            shutil.copyfile(url_or_path, local_file, follow_symlinks=False)
            return file_path_to_database_uri(local_file, database_dir)
        else:
            return file_path_to_database_uri(url_or_path, database_dir)
    else:
        if store_data_locally:
            return download_to_local_file(database_dir=database_dir, local_file=local_file, url=url_or_path)
        else:
            return url_or_path
