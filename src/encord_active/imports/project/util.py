import datetime
import uuid
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from encord_active.db.local_data import (
    db_uri_to_local_file_path,
    open_database_uri_image,
)
from encord_active.db.models import ProjectDataMetadata, ProjectDataUnitMetadata
from encord_active.imports.util import get_mimetype


def data_du_meta_for_local_image(
    database_dir: Path,
    project_hash: uuid.UUID,
    dataset_hash: uuid.UUID,
    dataset_title: str,
    data_hash: uuid.UUID,
    timestamp: datetime.datetime,
    data_title: str,
    data_uri: str,
    width: Optional[int],
    height: Optional[int],
    objects: list,
    classifications: list,
    object_answers: dict,
    classification_answers: dict,
) -> Tuple[ProjectDataMetadata, ProjectDataUnitMetadata]:
    local_uri = db_uri_to_local_file_path(data_uri, database_dir)
    download_image: Optional[Image.Image] = None
    if local_uri is not None:
        data_type: str = get_mimetype(local_uri, fallback="image")
    else:
        download_image = open_database_uri_image(data_uri, database_dir)
        data_type = download_image.format

    if width is not None and height is not None:
        img_width: int = width
        img_height: int = height
    elif local_uri is not None:
        img = Image.open(local_uri)
        img_width = img.width
        img_height = img.height
    elif download_image is not None:
        img_width = download_image.width
        img_height = download_image.height
    else:
        raise RuntimeError("Impossible case reached")

    data_meta = ProjectDataMetadata(
        project_hash=project_hash,
        data_hash=data_hash,
        label_hash=uuid.uuid4(),
        dataset_hash=dataset_hash,
        num_frames=1,
        frames_per_second=None,
        dataset_title=dataset_title,
        data_title=data_title,
        data_type="image",
        created_at=timestamp,
        last_edited_at=timestamp,
        object_answers=object_answers,
        classification_answers=classification_answers,
    )
    du_meta = ProjectDataUnitMetadata(
        project_hash=project_hash,
        du_hash=data_hash,
        frame=0,
        data_hash=data_hash,
        width=img_width,
        height=img_height,
        data_uri=data_uri,
        data_uri_is_video=False,
        data_title=data_title,
        data_type=data_type,
        objects=objects,
        classifications=classifications,
    )

    return data_meta, du_meta
