import contextlib
import logging
from io import BytesIO
from pathlib import Path
from typing import Generator, Optional, Union
from urllib.parse import unquote, urlparse

import av
import requests
from PIL import Image

from encord_active.db.models import ProjectDataUnitMetadata


def db_uri_to_local_file_path(database_uri: str, database_dir: Path) -> Optional[Path]:
    """
    Convert a data_uri value in the database into a local Path if one is referenced.
    """
    if database_uri.startswith(("/", "./", "../", "~/")):
        return Path(database_uri)
    if database_uri.startswith("file:"):
        return Path(unquote(urlparse(database_uri).path))
    if database_uri.startswith("relative://"):
        relative_path = database_uri[len("relative://") :]
        return database_dir / Path(relative_path)
    if database_uri.startswith("absolute://"):
        absolute_path = database_uri[len("absolute:/") :]
        return Path(absolute_path)
    return None


def file_path_to_database_uri(path: Path, project_dir: Path, relative: Optional[bool] = None) -> str:
    """
    Convert a local file path to a database_uri. This supports multiple formats including absolute & reltive
    addressing modes.
    """
    path = path.expanduser().absolute().resolve()
    if relative is not None and not relative:
        return path.as_uri()

    # Attempt to create relative path
    root_path = project_dir.expanduser().absolute().resolve()
    try:
        rel_path = path.relative_to(root_path)
        return f"relative://{rel_path.as_posix()}"
    except ValueError:
        if relative is not None:
            raise

    # Fallback to strict uri
    return path.as_uri()


def download_remote_to_bytes(url: str) -> bytes:
    r = requests.get(url)
    if r.status_code != 200:
        raise ConnectionError(f"Something happened, couldn't download file from: {url}")
    return r.content


def download_remote_to_file(url: str, file_path: Path) -> None:
    with open(file_path, "xb") as file:
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise ConnectionError(f"Something happened, couldn't download file from: {url}")
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
        file.flush()


def open_database_uri_image(database_uri: str, database_dir: Path) -> Image.Image:
    path = db_uri_to_local_file_path(database_uri, database_dir)
    if path is not None:
        return Image.open(path)
    img_bytes = download_remote_to_bytes(database_uri)
    return Image.open(BytesIO(img_bytes))


logger: logging.Logger = logging.getLogger("local-data")


@contextlib.contextmanager
def pyav_video_open(
    url_or_path: Union[str, Path], data_unit: ProjectDataUnitMetadata
) -> Generator[av.container.input.InputContainer, None, None]:
    try:
        # Try auto-detect video format
        with av.open(str(url_or_path), mode="r") as container:
            yield container
        return
    except ValueError as no_container_format:
        if "no container format" not in str(no_container_format):
            raise no_container_format
    logging.warning("Failed to infer video type, retrying with data_type")

    try:
        # Try with explicit data unit data type.
        with av.open(str(url_or_path), mode="r", format=data_unit.data_type) as container:
            yield container
        return
    except ValueError as no_container_format:
        if "no container format" not in str(no_container_format):
            raise no_container_format
    logging.warning("Failed to infer video type, retrying with data_title suffix")

    # Guess from file extension recorded in data title
    with av.open(str(url_or_path), mode="r", format=data_unit.data_title.split(".")[-1]) as container:
        yield container
