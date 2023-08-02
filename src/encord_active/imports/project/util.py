import shutil
import uuid
from pathlib import Path
from typing import Union

import requests

from encord_active.lib.common.data_utils import file_path_to_url


def get_data_uri(
    url_or_path: Union[str, Path],
    store_data_locally: bool,
    store_symlinks: bool,
    database_dir: Path,
) -> str:
    local_data_folder = database_dir / "local-data"
    local_data_folder.mkdir(exist_ok=True, parents=False)
    local_file = local_data_folder / str(uuid.uuid4())
    if local_file.exists():
        raise ValueError(f"uuid collision in local-data: {local_file.name}")
    if isinstance(url_or_path, Path):
        if store_data_locally and store_symlinks:
            local_file.symlink_to(url_or_path)
            return file_path_to_url(local_file, database_dir)
        elif store_data_locally:
            shutil.copyfile(url_or_path, local_file, follow_symlinks=False)
            return file_path_to_url(local_file, database_dir)
        else:
            return file_path_to_url(url_or_path, database_dir)
    else:
        if store_data_locally:
            with open(local_file, "xb") as file:
                r = requests.get(url_or_path, stream=True)
                if r.status_code != 200:
                    raise ConnectionError(f"Something happened, couldn't download file from: {url_or_path}")
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)
                file.flush()
            return file_path_to_url(local_file, database_dir)
        else:
            return url_or_path
