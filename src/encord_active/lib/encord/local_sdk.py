"""
This file mimics or extends some of the central structures in the encord SDK
like the::

    encord.Dataset
    encord.EncordUserClient
    encord.Project
    encord.dataset.DataRow
    encord.ontology.OntologyStructure
    encord.orm.label_row.LabelRowMetadata

to allow imports to be entirely local and not depend on communicating with the
Encord platform.
"""

import json
import mimetypes
import os
import random
import shutil
import string
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from uuid import uuid4

from encord.exceptions import (
    FileTypeNotSupportedError as EncordFiletypeNotSupportedError,
)
from encord.http.utils import CloudUploadSettings
from encord.ontology import OntologyStructure
from encord.orm.dataset import DataType
from encord.orm.label_row import LabelRow, LabelRowMetadata
from encord.project import AnnotationTaskStatus, LabelStatus
from PIL import Image
from prisma import Batch

from encord_active.lib.common.data_utils import file_path_to_url
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.project import ProjectFileStructure


class FileTypeNotSupportedError(EncordFiletypeNotSupportedError):
    ...


@dataclass(frozen=True)
class Dimensions:
    height: int
    width: int


@dataclass
class DataRowMedia:
    path: Path
    uid: str

    def __post_init__(self):
        self.path = self.path.resolve()


@dataclass
class LocalDataRow:
    uid: str
    label_hash: str
    title: str
    data_type: DataType
    media: List[DataRowMedia]
    created_at: datetime = field(default_factory=datetime.now)


def get_mimetype(path: Path) -> str:
    guess = mimetypes.guess_type(path)[0]
    if guess:
        return guess
    return f"unknown/{path.suffix[1:]}"


def get_dimensions(path, data_type: DataType) -> Dimensions:
    """
    Gets the dimensionality of the image.
    Note that PIL.Image has this nice ability to not read the entire image into
    memory to get the image size. This is much faster than, e.g., `cv2.imread`.

    Args:
        dr (LocalDataRow): the data row which points to the file to determine
            size from.

    Returns:
        The size of the image.
    """
    if data_type in {DataType.IMAGE, DataType.IMG_GROUP}:
        size = Image.open(path).size
        return Dimensions(size[1], size[0])
    else:
        raise FileTypeNotSupportedError("Video support for local initialisation is not implemented yet.")
        # with av.open(path) as container:
        #     stream = container.streams.video[0]
        #     for frame in container.decode(stream):
        #         width, height = frame.width, frame.height
        #         return Dimensions(width=width, height=height)


def get_data_units(dr: LocalDataRow) -> Dict[str, dict]:
    data_units: Dict[str, dict] = {}
    for i, media in enumerate(dr.media):
        title = media.path.name if len(dr.media) > 1 else dr.title
        try:
            dims = get_dimensions(media.path, dr.data_type)
        except NotImplementedError as e:
            print(str(e))
            print(f"Skipping `{media.path}`")
            continue

        data_units[media.uid] = {
            "data_hash": media.uid,
            "data_title": title,
            "data_type": get_mimetype(media.path),
            "data_sequence": i,
            "labels": {"objects": [], "classifications": []},
            "data_link": media.path.as_posix(),
            "width": dims.width,
            "height": dims.height,
        }
    return data_units


def get_empty_label_row(meta: LabelRowMetadata, dr: LocalDataRow, dataset_title: str) -> LabelRow:
    """
    Constructs an empty label row (without annotations) based of the information given.
    """
    if dr.data_type != DataType.IMAGE:
        raise NotImplementedError("Not implemented as COCO only uses single images")

    data_units = get_data_units(dr)
    return LabelRow(
        {
            "label_hash": meta.label_hash,
            "dataset_hash": meta.dataset_hash,
            "dataset_title": dataset_title,
            "data_title": meta.data_title,
            "data_hash": meta.data_hash,
            "data_type": meta.data_type,
            "data_units": data_units,
            "object_answers": {},
            "classification_answers": {},
            "object_actions": {},
            "label_status": meta.label_status,
        }
    )


class LocalDataset:
    """
    Mimics the `encord.dataset` but without interacting with the Encord platform.
    """

    def __init__(
        self,
        title: str,
        dataset_hash: str,
        data_path: Path,
        use_symlinks: bool,
        project_file_structure: ProjectFileStructure,
    ):
        self.title: str = title
        self.dataset_hash: str = dataset_hash
        self.data_path: Path = data_path
        self.use_symlinks: bool = use_symlinks

        self._data_rows: Dict[str, LocalDataRow] = {}
        self._project_fs = project_file_structure

    @property
    def data_rows(self) -> List[LocalDataRow]:
        return list(self._data_rows.values())

    def get_data_row(self, data_hash: str) -> LocalDataRow:
        """
        This function is not available in the `encord.Dataset` object but
        added here for ease of implementation
        """
        if data_hash not in self._data_rows:
            raise ValueError(f"Data hash `{data_hash}` not in the dataset")
        return self._data_rows[data_hash]

    @staticmethod
    def check_mime_type(path: Path):
        if "image" not in get_mimetype(path):
            raise FileTypeNotSupportedError("Video support for local initialisation is not implemented yet.")

    def create_image_group(
        self,
        file_paths: Iterable[Union[str, Path]],
        max_workers: Optional[int] = None,
        cloud_upload_settings: CloudUploadSettings = CloudUploadSettings(),
        title: Optional[str] = None,
    ):
        """
        Replication of the encord.Dataset class.

        NOTE: The function declaration replicates the Encord dataset, so some
        parameters are ignored.

        Args:
            file_paths: The ordered paths to where the files for the image group are stored.
            max_workers: !THIS IS IGNORED!
            cloud_upload_settings: !THIS IS IGNORED!
            title: The title of the image group.
        """
        _file_paths = list(map(Path, file_paths))
        [self.check_mime_type(p) for p in _file_paths]  # Raises FileTypeNotSupportedError

        data_hash = str(uuid4())
        label_hash = str(uuid4())

        out_dir = self.data_path / label_hash / "images"
        out_dir.mkdir(exist_ok=True, parents=True)

        media: List[DataRowMedia] = []
        for _uri in _file_paths:
            uid = str(uuid4())
            out_file = out_dir / f"{uid}{_uri.suffix}"
            if self.use_symlinks:
                os.symlink(_uri.expanduser().absolute(), out_file)
            else:
                shutil.copy(_uri, out_file)
            media.append(DataRowMedia(_uri, uid))

        if not title:
            uid = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
            title = f"image-group-{uid}"

        data_row = LocalDataRow(
            uid=data_hash, label_hash=label_hash, title=title, data_type=DataType.IMAGE, media=media
        )
        self._data_rows[data_hash] = data_row

    def upload_image(self, file_path: Union[Path, str], title: str = "") -> LocalDataRow:
        """
        Copies image to `self.data_path/label_hash/images/datahash.ext`.

        Note that Encord Dataset also support adding image groups and videos.
        However, for the COCO import, which is the only current use case, only
        single images are used, so only this is implemented.

        Args:
            file_path (Union[Path, str]): The image to add.
            title (str): The title of the image to be associated with the image.
        """
        if isinstance(file_path, str):
            _uri = Path(file_path)
        else:
            _uri = file_path

        self.check_mime_type(_uri)

        data_hash = str(uuid4())
        label_hash = str(uuid4())

        out_dir = self.data_path / label_hash / "images"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = out_dir / f"{data_hash}{_uri.suffix}"

        if self.use_symlinks:
            os.symlink(_uri.expanduser().absolute(), out_file)
        else:
            shutil.copy(_uri, out_file)
        dr_media = DataRowMedia(_uri, data_hash)

        if not title:
            title = _uri.name

        data_row = LocalDataRow(
            uid=data_hash, label_hash=label_hash, title=title, data_type=DataType.IMAGE, media=[dr_media]
        )
        self._data_rows[data_hash] = data_row
        return data_row


@dataclass
class LocalOntology:
    """
    Mimics the `encord.ontology.Ontology` but without the need of a Querier, etc.
    """

    ontology_hash: str
    title: str
    description: str
    structure: OntologyStructure


class LocalProject:
    """
    Mimics `encord.Project` but without interacting with the Encord platform.
    """

    def __init__(
        self,
        data_path: Path,
        title: str,
        description: str,
        ontology: LocalOntology,
        datasets: List[LocalDataset],
        project_file_structure: ProjectFileStructure,
    ):
        self.data_path: Path = data_path
        self.title: str = title
        self.description: str = description
        self._ontology: LocalOntology = ontology
        self._datasets: Dict[str, LocalDataset] = {d.dataset_hash: d for d in datasets}

        self.project_hash: str = str(uuid4())
        self._label_row_meta: Dict[str, LabelRowMetadata] = {}
        self._label_rows: Dict[str, LabelRow] = {}
        self._dh_to_lh: Dict[str, str] = {}

        self._project_file_structure = project_file_structure

        self._populate_label_row_meta()

    @property
    def ontology(self) -> dict:
        return self._ontology.structure.to_dict()

    @property
    def label_row_meta(self) -> List[LabelRowMetadata]:
        return list(self._label_row_meta.values())

    @property
    def label_rows(self) -> List[LabelRow]:
        return list(self._label_rows.values())

    def _populate_label_row_meta(self):
        for dataset in self._datasets.values():
            for dr in dataset.data_rows:
                for du in get_data_units(dr).values():
                    meta = LabelRowMetadata(
                        label_hash=dr.label_hash,
                        dataset_hash=dataset.dataset_hash,
                        dataset_title=dataset.title,
                        data_hash=du["data_hash"],
                        data_type=du["data_type"],
                        data_title=du["data_title"],
                        data_link=du["data_link"],
                        height=du["height"],
                        width=du["width"],
                        annotation_task_status=AnnotationTaskStatus.QUEUED,
                        workflow_graph_node=None,
                        label_status=LabelStatus.NOT_LABELLED,
                        created_at=dr.created_at or datetime.now(),
                        last_edited_at=dr.created_at or datetime.now(),
                        is_shadow_data=False,
                        number_of_frames=1,
                        duration=None,
                        frames_per_second=None,
                    )
                    self._label_row_meta[dr.label_hash] = meta
                    self._dh_to_lh[dr.uid] = dr.label_hash

    def create_label_row(self, data_hash: str):
        label_hash = self._dh_to_lh.get(data_hash)
        if not label_hash:
            raise ValueError("Data hash not associated to project")

        meta: LabelRowMetadata = self._label_row_meta[label_hash]
        dataset = self._datasets[meta.dataset_hash]
        data_row = dataset.get_data_row(meta.data_hash)
        self._label_rows[label_hash] = get_empty_label_row(meta, data_row, dataset.title)
        return self._label_rows[label_hash]

    def get_label_row(self, label_hash, get_signed_url: bool = False) -> LabelRow:
        if get_signed_url:
            raise ValueError(
                "This is a local project without connection to any remote db, so "
                "no download url available. The local path can be found under the data units data links"
            )
        return self._label_rows[label_hash]

    def save_label_row(self, uid: str, label: LabelRow, batch: Optional[Batch] = None):
        label_hash: str = uid
        if uid not in self._label_rows:
            raise ValueError("No label row with that uid. Call `LocalProject.create_label_row` first.")

        self._label_rows[label_hash] = label

        raw_label = dict(label)
        raw_label["data_type"] = raw_label["data_type"].split("/")[0]
        label_row_json = json.dumps(raw_label)
        default_timestamp = "0"

        connection = batch if batch is not None else PrismaConnection(self._project_file_structure)

        with connection as conn:  # type: ignore
            conn.labelrow.upsert(
                where={"data_hash": label.data_hash},
                data={
                    "create": {
                        "label_hash": label.label_hash,
                        "data_hash": label.data_hash,
                        "data_title": label.data_title,
                        "data_type": label.data_type.split("/")[0],
                        "created_at": label.get("created_at", default_timestamp),
                        "last_edited_at": label.get("last_edited_at", default_timestamp),
                        "label_row_json": label_row_json,
                    },
                    "update": {
                        "label_hash": label.label_hash,
                        "data_title": label.data_title,
                        "last_edited_at": label.get("last_edited_at", default_timestamp),
                        "label_row_json": label_row_json,
                    },
                },
            )

            def find_image_path(data_unit) -> str:
                path = self.data_path / label_hash / "images"

                image_path = next(path.glob(f"{data_unit['data_hash']}.*"))
                return str(image_path)

            for du_hash, data_unit in label.data_units.items():
                image_path = Path(find_image_path(data_unit)).absolute()
                image = Image.open(image_path)
                conn.dataunit.create(
                    data={
                        "data_hash": du_hash,
                        "data_title": label.data_title,
                        "frame": 0,  # FIXME: only used as images currently, will break for image groups and videos
                        "data_uri": file_path_to_url(image_path, self._project_file_structure.project_dir),
                        "lr_data_hash": label.data_hash,
                        "width": image.width,
                        "height": image.height,
                        "fps": 0,  # FIXME: only used as images currently, videos not supported for local_sdk
                    }
                )


class LocalUserClient:
    """
    Mimics the `encord.EncordUserClient` but without interacting with the Encord
    platform.

    Although it's created in a way that supports multiple datasets and project,
    it is currently only intended for usage with one of each.
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.data_path = project_path / "data"
        self.datasets: Dict[str, LocalDataset] = {}
        self.ontologies: Dict[str, LocalOntology] = {}
        self.projects: Dict[str, LocalProject] = {}
        self.project_fs = ProjectFileStructure(project_path)

    def create_dataset(self, title: str, use_symlinks: bool = False) -> LocalDataset:
        dataset_hash = str(uuid4())
        dataset = LocalDataset(
            title,
            dataset_hash=dataset_hash,
            data_path=self.data_path,
            use_symlinks=use_symlinks,
            project_file_structure=self.project_fs,
        )
        self.datasets[dataset_hash] = dataset
        return dataset

    def get_dataset(self, dataset_hash: str) -> LocalDataset:
        return self.datasets[dataset_hash]

    def create_ontology(self, title: str, description: str, structure: OntologyStructure) -> LocalOntology:
        uid = str(uuid4())
        ontology = LocalOntology(uid, title, description, structure)
        self.ontologies[uid] = ontology
        return ontology

    def get_ontology(self, ontology_hash: str) -> LocalOntology:
        return self.ontologies[ontology_hash]

    def create_project(
        self, project_title: str, dataset_hashes: List[str], ontology_hash: str, description: str = ""
    ) -> LocalProject:
        ontology = self.get_ontology(ontology_hash)
        datasets = list(map(self.get_dataset, dataset_hashes))

        project = LocalProject(
            data_path=self.data_path,
            title=project_title,
            description=description,
            ontology=ontology,
            datasets=datasets,
            project_file_structure=self.project_fs,
        )
        self.projects[project.project_hash] = project
        return project
