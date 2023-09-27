"""
Public API for accessing an encord active project.

This provides stable interfaces for common operations on encord active projects.

This is for external use only.

Currently experimental.
"""
import copy
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import av
import encord
from encord import EncordUserClient
from encord.http.constants import RequestsSettings
from encord.objects import OntologyStructure
from PIL import Image
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.cli.app_config import app_config
from encord_active.db.local_data import db_uri_to_local_file_path
from encord_active.db.models import (
    Project,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    get_engine,
)

__all__ = ["ActiveContext", "ActiveAnnotatedFrame"]


class ActiveAnnotatedFrame:
    """
    Stable API reference to an active annotated image.
    """

    _image: Image.Image
    _data_unit: ProjectDataUnitMetadata
    _data: ProjectDataMetadata

    def __init__(
        self,
        data_unit: ProjectDataUnitMetadata,
        data: ProjectDataMetadata,
        image: Image,
    ) -> None:
        self._image = image
        self._data_unit = data_unit
        self._data = data

    @property
    def image(self) -> Image.Image:
        return self._image

    @property
    def objects(self) -> list:
        return copy.deepcopy(self._data_unit.objects)

    @property
    def classifications(self) -> list:
        return copy.deepcopy(self._data_unit.classifications)

    @property
    def object_answers(self) -> dict:
        return copy.deepcopy(self._data.object_answers)

    @property
    def classification_answers(self) -> dict:
        return copy.deepcopy(self._data.classification_answers)


class ActiveProject:
    _engine: Engine
    _database_dir: Path
    _encord_client: EncordUserClient
    _encord_project: encord.Project

    def __init__(self, engine: Engine, database_dir: Path, ssh_key: str, project: Project) -> None:
        self._engine = engine
        self._database_dir = database_dir
        self._project = project
        self._encord_client = EncordUserClient.create_with_ssh_private_key(
            ssh_key,
            requests_settings=RequestsSettings(max_retries=5),
        )
        self._encord_project = self._encord_client.get_project(str(self._project.project_hash))

    @property
    def project_hash(self) -> uuid.UUID:
        return self._project.project_hash

    @property
    def name(self) -> str:
        return self._project.name

    @property
    def description(self) -> str:
        return self._project.description

    @property
    def ontology(self) -> OntologyStructure:
        return OntologyStructure.from_dict(copy.deepcopy(self._project.ontology))

    def _lookup_encord_url(
        self,
        data_hash: uuid.UUID,
    ) -> Dict[uuid.UUID, str]:
        video, images = self._encord_project.get_data(str(data_hash), get_signed_url=True)
        encord_url_dict = {}
        for image in images or []:
            encord_url_dict[uuid.UUID(image["data_hash"])] = str(image["file_link"])
        if video is not None:
            encord_url_dict[data_hash] = str(video["file_link"])
        return encord_url_dict

    def _lookup_url(self, data_hash: uuid.UUID, du_hash: uuid.UUID, db_uri: Optional[str]) -> Union[str, Path]:
        if db_uri is None:
            return self._lookup_encord_url(data_hash)[du_hash]
        else:
            url_path = db_uri_to_local_file_path(db_uri, self._database_dir)
            if url_path is not None:
                return url_path
            else:
                return db_uri

    def iter_frames(self) -> Iterator[ActiveAnnotatedFrame]:
        with Session(self._engine) as sess:
            all_data_units = sess.exec(
                select(ProjectDataUnitMetadata)
                .where(ProjectDataUnitMetadata.project_hash == self.project_hash)
                .order_by(
                    ProjectDataUnitMetadata.data_hash, ProjectDataUnitMetadata.du_hash, ProjectDataUnitMetadata.frame
                )
            ).fetchall()
            all_data = sess.exec(
                select(ProjectDataMetadata).where(ProjectDataMetadata.project_hash == self.project_hash)
            )
            data_hash_map = {d.data_hash: d for d in all_data}
        data_unit_iter = iter(all_data_units)
        while True:
            data_unit = next(data_unit_iter)
            url = self._lookup_url(data_unit.data_hash, data_unit.du_hash, data_unit.data_uri)
            if data_unit.data_uri_is_video:
                with av.open(url) as container:
                    video_decode_iter = iter(container.decode(video=0))
                    frame0 = next(video_decode_iter)
                    if frame0.frame != 0:
                        raise ValueError(f"Video starts at frame: {frame0.frame} != 0")
                    if frame0.is_corrupt:
                        raise ValueError(f"Corrupt video frame: {frame0.frame}")
                    yield ActiveAnnotatedFrame(
                        data_unit, data_hash_map[data_unit.data_hash], frame0.to_image().convert("RGB")
                    )
                    frame_idx = 0
                    while True:
                        frame_idx += 1
                        frame = next(video_decode_iter, None)
                        if frame is None:
                            break
                        frame_data_unit: ProjectDataUnitMetadata = next(data_unit_iter)
                        if frame_data_unit.data_hash != data_unit.data_hash:
                            raise ValueError(
                                f"Video decode length mismatch: {data_unit.data_hash}: {frame_idx} not found in db"
                            )
                        if frame.is_corrupt:
                            raise ValueError(f"Corrupt video frame: {frame_idx}")
                        if frame_data_unit.frame != frame_idx:
                            raise ValueError(f"Out of order frame iteration: {frame_data_unit.frame} != {frame_idx}")
                        yield ActiveAnnotatedFrame(
                            frame_data_unit, data_hash_map[frame_data_unit.data_hash], frame.to_image().convert("RGB")
                        )
            else:
                image = Image.open(url)
                yield ActiveAnnotatedFrame(data_unit, data_hash_map[data_unit.data_hash], image)


class ActiveContext:
    _database_dir: Path
    _engine: Engine
    _ssh_key: str

    def __init__(self, path: Path, ssh_key: Optional[str] = None) -> None:
        self._database_dir = path
        self._engine = get_engine(path / "encord_active.sqlite")
        if ssh_key is None:
            self._ssh_key = app_config.get_or_query_ssh_key().read_text("utf-8")
        else:
            self._ssh_key = ssh_key

    def list_projects(self) -> List[ActiveProject]:
        with Session(self._engine) as sess:
            all_projects = sess.exec(select(Project)).fetchall()
        return [ActiveProject(self._engine, self._database_dir, self._ssh_key, project) for project in all_projects]

    def get_project(self, project_hash: uuid.UUID) -> Optional[ActiveProject]:
        with Session(self._engine) as sess:
            project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            return None
        return ActiveProject(self._engine, self._database_dir, self._ssh_key, project)
