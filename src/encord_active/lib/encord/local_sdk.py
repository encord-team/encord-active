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
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, TypedDict, Union
from uuid import uuid4

from encord.dataset import DataRow
from encord.ontology import OntologyStructure
from encord.orm.dataset import DataType
from encord.orm.label_row import LabelRow
from encord.project import AnnotationTaskStatus, LabelStatus
from PIL import Image


@dataclass(frozen=True)
class Dimensions:
    height: int
    width: int


class LabelRowMetadata(TypedDict):
    """
    This is a class that mimics the encord.orm.label_row.LabelRowMetadata
    The reason why we don't use the actual class is because we cannot update
    fields in the original class.
    """

    label_hash: str
    data_hash: str
    dataset_hash: str
    data_title: str
    data_type: str
    label_status: str
    annotation_task_status: str


class LocalDataRow(DataRow):
    def __init__(
        self,
        uid: str,
        label_hash: str,
        title: str,
        data_type: DataType,
        path: Path,
    ):
        """
        Mimics the Encord DataRow but with two additional parameters `path` and
        `label_hash` to ease implementation.

        Args:
            uid (str): The data hash
            label_hash (str): The hash of the associated label row in the `LocalProject.
            title (str): The title of the Data row.
            data_type (DataType): The data type of the DataRow. Not really used as
                everything is `DataType.IMAGE` at the moment.
            path (Path): The local path to the data asset.
        """
        created_at: datetime = datetime.now()
        super(LocalDataRow, self).__init__(uid=uid, title=title, data_type=data_type, created_at=created_at)  # type: ignore

        self.path = path.resolve()
        self.label_hash = label_hash


def get_mimetype(dr: LocalDataRow) -> str:
    guess = mimetypes.guess_type(dr.path)[1]
    if guess:
        return guess
    return f"image/{dr.path.suffix[1:]}"


def get_dimensions(dr: LocalDataRow) -> Dimensions:
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
    assert dr.data_type == DataType.IMAGE, "Only single image support for now"
    size = Image.open(dr.path).size
    return Dimensions(size[1], size[0])


def get_empty_label_row(meta: LabelRowMetadata, dr: LocalDataRow, dataset_title: str) -> LabelRow:
    """
    Constructs an empty label row (without annotations) based of the information given.
    """
    if dr.data_type != DataType.IMAGE:
        raise NotImplementedError("Not implemented as COCO only uses single images")

    dims = get_dimensions(dr)
    return LabelRow(
        {
            "label_hash": meta["label_hash"],
            "dataset_hash": meta["dataset_hash"],
            "dataset_title": dataset_title,
            "data_title": meta["data_title"],
            "data_type": "image",
            "data_units": {
                dr.uid: {
                    "data_hash": dr.uid,
                    "data_title": dr.title,
                    "data_type": get_mimetype(dr),
                    "data_sequence": 0,
                    "labels": {"objects": [], "classifications": []},
                    "data_link": dr.path.as_posix(),
                    "width": dims.width,
                    "height": dims.height,
                }
            },
            "object_answers": {},
            "classification_answers": {},
            "object_actions": {},
            "label_status": meta["label_status"],
        }
    )


class LocalDataset:
    """
    Mimics the `encord.dataset` but without interacting with the Encord platform.
    """

    def __init__(self, title: str, dataset_hash: str, data_path: Path, use_symlinks: bool):
        self.title: str = title
        self.dataset_hash: str = dataset_hash
        self.data_path: Path = data_path
        self.use_symlinks: bool = use_symlinks

        self._data_rows: Dict[str, LocalDataRow] = {}

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

    def upload_image(self, file_path: Union[Path, str], title: str = ""):
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

        data_hash = str(uuid4())
        label_hash = str(uuid4())

        out_dir = self.data_path / label_hash / "images"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_file = out_dir / f"{data_hash}{_uri.suffix}"

        if self.use_symlinks:
            os.symlink(_uri.expanduser().absolute(), out_file)
        else:
            shutil.copy(_uri, out_file)

        if not title:
            title = _uri.name

        data_row = LocalDataRow(
            uid=data_hash, label_hash=label_hash, title=title, data_type=DataType.IMAGE, path=out_file
        )
        self._data_rows[data_hash] = data_row


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
        self, data_path: Path, title: str, description: str, ontology: LocalOntology, datasets: List[LocalDataset]
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

        self._populate_label_row_meta()

    @property
    def ontology(self) -> dict:
        return self._ontology.structure.to_dict()

    @property
    def label_rows(self) -> List[LabelRowMetadata]:
        return list(self._label_row_meta.values())

    def _populate_label_row_meta(self):
        for dataset in self._datasets.values():
            for dr in dataset.data_rows:
                meta = LabelRowMetadata(
                    label_hash=dr.label_hash,
                    dataset_hash=dataset.dataset_hash,
                    data_hash=dr.uid,
                    data_type=dr.data_type.to_upper_case_string(),
                    data_title=dr.title,
                    annotation_task_status=AnnotationTaskStatus.QUEUED.value,
                    label_status=LabelStatus.NOT_LABELLED.value,
                    # is_shadow_data=False,
                )
                self._label_row_meta[dr.label_hash] = meta
                self._dh_to_lh[dr.uid] = dr.label_hash

    def create_label_row(self, data_hash: str):
        label_hash = self._dh_to_lh.get(data_hash)
        if not label_hash:
            raise ValueError("Data hash not associated to project")

        meta: LabelRowMetadata = self._label_row_meta[label_hash]
        dataset = self._datasets[meta["dataset_hash"]]
        data_row = dataset.get_data_row(meta["data_hash"])
        self._label_rows[label_hash] = get_empty_label_row(meta, data_row, dataset.title)
        return self._label_rows[label_hash]

    def get_label_row(self, label_hash, get_signed_url: bool = False) -> LabelRow:
        if get_signed_url:
            raise ValueError(
                "This is a local project without connection to any remote db, so "
                "no download url available. The local path can be found under the data units data links"
            )
        return self._label_rows[label_hash]

    def save_label_row(self, uid: str, label: LabelRow):
        label_hash: str = uid
        if uid not in self._label_rows:
            raise ValueError("No label row with that uid. Call `LocalProject.create_label_row` first.")

        meta: LabelRowMetadata = self._label_row_meta[label_hash]
        meta["label_status"] = label["label_status"]

        self._label_rows[label_hash] = label

        label_row_path = self.data_path / uid / "label_row.json"
        with label_row_path.open("w") as f:
            json.dump(label, f, indent=2)


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

    def create_dataset(self, title: str, use_symlinks: bool = False) -> LocalDataset:
        dataset_hash = str(uuid4())
        dataset = LocalDataset(title, dataset_hash=dataset_hash, data_path=self.data_path, use_symlinks=use_symlinks)
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
        )
        self.projects[project.project_hash] = project
        return project
