from __future__ import annotations

import itertools
import json
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import yaml
from encord import Project as EncordProject
from encord.constants.enums import DataType
from encord.objects.ontology_structure import OntologyStructure
from encord.orm.label_row import LabelRow
from encord.project import LabelRowMetadata
from loguru import logger

from encord_active.cli.config import app_config
from encord_active.lib.common.utils import (
    collect_async,
    download_file,
    slice_video_into_frames,
    try_execute,
)
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.db.prisma_init import ensure_prisma_db
from encord_active.lib.encord.local_sdk import handle_enum_and_datetime
from encord_active.lib.encord.utils import get_client
from encord_active.lib.project.metadata import (
    ProjectMeta,
    ProjectNotFound,
    fetch_project_meta,
)
from encord_active.lib.project.project_file_structure import ProjectFileStructure

logger = logger.opt(colors=True)
encord_logger = logging.getLogger("encord")
encord_logger.setLevel(logging.ERROR)


class Project:
    def __init__(self, project_dir: Union[str, Path]):
        self.file_structure = ProjectFileStructure(project_dir)
        self.project_meta: ProjectMeta
        try:
            self.project_meta = fetch_project_meta(self.file_structure.project_dir)
        except ProjectNotFound:
            self.project_meta = ProjectMeta(
                project_description="", project_hash="", project_title="", ssh_key_path="", has_remote=False
            )
        self.project_hash: str = ""
        self.ontology: OntologyStructure = OntologyStructure.from_dict(dict(objects=[], classifications=[]))
        self.label_row_metas: Dict[str, LabelRowMetadata] = {}
        self.label_rows: Dict[str, LabelRow] = {}
        self.image_paths: Dict[str, Dict[str, Path]] = {}

        ensure_prisma_db(self.file_structure.prisma_db)

    def load(self, subset_size: Optional[int] = None) -> Project:
        """
        Initialize a project from local storage.

        :param subset_size: The number of label rows to fetch. If None then all label rows are fetched.
        :return: The updated project instance.
        """
        if self.is_loaded:
            return self

        if not self.file_structure.project_dir.exists():
            raise FileNotFoundError(f"`{self.file_structure.project_dir}` does not exist")
        if not self.file_structure.project_dir.is_dir():
            raise NotADirectoryError(f"`{self.file_structure.project_dir}` does not point to a directory")

        self.project_meta = fetch_project_meta(self.file_structure.project_dir)
        self.project_hash = self.project_meta["project_hash"]

        self.__load_ontology()
        self.__load_label_row_meta(subset_size)
        self.__load_label_rows()

        return self

    def from_encord_project(self, encord_project: EncordProject) -> Project:
        """
        Initialize a project from Encord platform.

        :param encord_project: Encord project from where the data is fetched.
        :return: The updated project instance.
        """
        if self.is_loaded:
            return self

        self.file_structure.project_dir.mkdir(parents=True, exist_ok=True)

        # todo enforce clean up when we are sure it won't impact performance in other sections (like PredictionWriter)
        # also don't forget to add `from shutil import rmtree` at the top (pylint tags it as unused right now)
        # clean project_dir content
        # for path in self.file_structure.project_dir.iterdir():
        #     if path.is_file():
        #         path.unlink()
        #     elif path.is_dir():
        #         rmtree(path)
        self.encord_project = encord_project

        self.__save_project_meta(encord_project)
        self.save_ontology(OntologyStructure.from_dict(encord_project.ontology))
        self.__download_and_save_label_rows(encord_project)
        self.__save_label_row_meta(encord_project)

        return self.load()

    def refresh(self):
        """
        Refresh project data and labels using its remote project in Encord Annotate.

        :return: The updated project instance.
        """

        self.project_meta = fetch_project_meta(self.file_structure.project_dir)
        if not self.project_meta.get("has_remote", False):
            raise AttributeError("The project does not have a remote project associated to it.")

        project_hash = self.project_meta.get("project_hash")
        if project_hash is None:
            raise AttributeError("The project does not have a remote project associated to it.")

        ssh_key_path = self.project_meta.get("ssh_key_path", app_config.get_ssh_key())
        if ssh_key_path is None:
            raise AttributeError(
                "The project metadata is missing the path to the private SSH key needed to log into Encord Annotate. "
                f"Add such path to the property `ssh_key_path` in the file {self.file_structure.project_meta}."
            )

        encord_client = get_client(Path(ssh_key_path))
        encord_project = encord_client.get_project(project_hash)
        self.__download_and_save_label_rows(encord_project)
        self.__save_label_row_meta(encord_project)  # Update cached metadata of the label rows (after new data sync)

        return self.load()

    @property
    def is_loaded(self) -> bool:
        return all(
            map(
                bool,
                [
                    self.project_meta,
                    self.project_hash,
                    self.label_row_metas,
                    self.label_rows,
                ],
            )
        )

    def __save_project_meta(self, encord_project: EncordProject):
        project_meta_file_path = self.file_structure.project_meta
        self.project_meta.update(
            {
                "project_title": encord_project.title,
                "project_description": encord_project.description,
                "project_hash": encord_project.project_hash,
                "has_remote": True,
            }
        )
        project_meta_file_path.write_text(yaml.safe_dump(self.project_meta), encoding="utf-8")

    def save_ontology(self, ontology: OntologyStructure):
        ontology_file_path = self.file_structure.ontology
        ontology_file_path.write_text(json.dumps(ontology.to_dict(), indent=2), encoding="utf-8")

    def __load_ontology(self):
        ontology_file_path = self.file_structure.ontology
        if not ontology_file_path.exists():
            raise FileNotFoundError(f"Expected file `ontology.json` at {ontology_file_path.parent}")
        self.ontology = OntologyStructure.from_dict(json.loads(ontology_file_path.read_text(encoding="utf-8")))

    def __save_label_row_meta(self, encord_project: EncordProject):
        label_row_meta = {
            lrm.label_hash: handle_enum_and_datetime(lrm)
            for lrm in encord_project.list_label_rows()
            if lrm.label_hash is not None
        }
        for meta in label_row_meta.values():
            meta["created_at"] = meta["created_at"].rsplit(".", maxsplit=1)[0]
            meta["last_edited_at"] = meta["last_edited_at"].rsplit(".", maxsplit=1)[0]

        self.file_structure.label_row_meta.write_text(json.dumps(label_row_meta, indent=2), encoding="utf-8")

    def __load_label_row_meta(self, subset_size: Optional[int] = None) -> dict[str, LabelRowMetadata]:
        label_row_meta_file_path = self.file_structure.label_row_meta
        if not label_row_meta_file_path.exists():
            raise FileNotFoundError(f"Expected file `label_row_meta.json` at {label_row_meta_file_path.parent}")

        self.label_row_metas = {
            lr_hash: LabelRowMetadata.from_dict(self.__populate_label_row_metadata_defaults(lr_meta))
            for lr_hash, lr_meta in itertools.islice(
                json.loads(label_row_meta_file_path.read_text(encoding="utf-8")).items(), subset_size
            )
        }
        return self.label_row_metas

    def save_label_row(self, label_row: LabelRow):
        lr_structure = self.file_structure.label_row_structure(label_row["label_hash"])
        lr_structure.label_row_file.write_text(json.dumps(label_row, indent=2), encoding="utf-8")

    def __download_and_save_label_rows(self, encord_project: EncordProject):
        label_rows = self.__download_label_rows_and_data(encord_project, self.file_structure)
        split_lr_videos(label_rows, self.file_structure)
        logger.info("Data and labels successfully synced from the remote project")
        return

    def __download_label_rows_and_data(
        self,
        project: EncordProject,
        project_file_structure: ProjectFileStructure,
        filter_fn: Optional[Callable[..., bool]] = lambda x: x["label_hash"] is not None,
        subset_size: Optional[int] = None,
    ) -> List[LabelRow]:
        try:
            current_label_row_metas = self.__load_label_row_meta()
        except FileNotFoundError:
            current_label_row_metas = dict()

        latest_label_row_metas = [
            LabelRowMetadata.from_dict(self.__populate_label_row_metadata_defaults(lr_meta))
            for lr_meta in itertools.islice(filter(filter_fn, project.label_rows), subset_size)
        ]

        label_rows_to_download: list[str] = []
        label_rows_to_update: list[str] = []
        for label_row_meta in latest_label_row_metas:
            if label_row_meta.label_hash not in current_label_row_metas:
                label_rows_to_download.append(label_row_meta.label_hash)
            else:
                current_label_row_version_hash = current_label_row_metas[label_row_meta.label_hash].last_edited_at
                latest_label_row_version_hash = label_row_meta.last_edited_at
                if current_label_row_version_hash != latest_label_row_version_hash:
                    label_rows_to_update.append(label_row_meta.label_hash)

        # Update label row content
        if len(label_rows_to_update) > 0:
            collect_async(
                partial(
                    download_label_row,
                    project=project,
                    project_file_structure=project_file_structure,
                ),
                label_rows_to_update,
                desc="Updating label rows",
            )
        else:
            logger.info("No existent data needs to be updated.")

        # Download new project data
        if len(label_rows_to_download) > 0:
            downloaded_label_rows = collect_async(
                partial(
                    download_label_row_and_data,
                    project=project,
                    project_file_structure=project_file_structure,
                ),
                label_rows_to_download,
                desc="Collecting new data",
            )
        else:
            downloaded_label_rows = []
            logger.info("No new data to be downloaded.")

        return downloaded_label_rows

    def __load_label_rows(self):
        self.label_rows = {}
        self.image_paths = {}
        for lr_hash in self.label_row_metas.keys():
            lr_structure = self.file_structure.label_row_structure(lr_hash)
            if not lr_structure.label_row_file.is_file() or not lr_structure.images_dir.is_dir():
                logger.warning(
                    f"Skipping label row <blue>`{lr_hash}`</blue> as its content wasn't found in the storage."
                )
                continue
            self.label_rows[lr_hash] = LabelRow(json.loads(lr_structure.label_row_file.read_text(encoding="utf-8")))
            self.image_paths[lr_hash] = dict((du_file.stem, du_file) for du_file in lr_structure.images_dir.iterdir())

    def __populate_label_row_metadata_defaults(self, lr_dict: dict):
        img_dir = self.file_structure.label_row_structure(lr_dict["label_hash"]).images_dir
        image_pth = img_dir.as_posix() if img_dir.is_dir() else ""

        return {
            "data_link": image_pth,
            "dataset_title": "",
            "is_shadow_data": False,
            "number_of_frames": 1,
            **lr_dict,
        }


def download_label_row(
    label_hash: str, project: EncordProject, project_file_structure: ProjectFileStructure
) -> LabelRow:
    lr_structure = project_file_structure.label_row_structure(label_hash)
    lr_structure.path.mkdir(parents=True, exist_ok=True)
    label_row = try_execute(project.get_label_row, 5, {"uid": label_hash})
    lr_structure.label_row_file.write_text(json.dumps(label_row, indent=2), encoding="utf-8")
    with PrismaConnection(project_file_structure) as conn:
        conn.labelrow.upsert(
            where={"data_hash": label_row.data_hash},
            data={
                "create": {
                    "label_hash": label_row.label_hash,
                    "data_hash": label_row.data_hash,
                    "data_title": label_row.data_title,
                    "data_type": label_row.data_type,
                    "created_at": label_row.created_at,
                    "last_edited_at": label_row.last_edited_at,
                    "location": lr_structure.label_row_file.as_posix(),
                },
                "update": {
                    "label_hash": label_row.label_hash,
                    "data_title": label_row.data_title,
                    "created_at": label_row.created_at,  # don't update this field if it's set in unannotated data
                    "last_edited_at": label_row.last_edited_at,
                    "location": lr_structure.label_row_file.as_posix(),
                },
            },
        )

    return label_row


def download_data(label_row: LabelRow, project_file_structure: ProjectFileStructure):
    lr_structure = project_file_structure.label_row_structure(label_row.label_hash)
    lr_structure.images_dir.mkdir(parents=True, exist_ok=True)
    data_units = sorted(label_row.data_units.values(), key=lambda du: int(du["data_sequence"]))
    for du in data_units:
        suffix = f".{du['data_type'].split('/')[1]}"
        destination = (lr_structure.images_dir / du["data_hash"]).with_suffix(suffix)
        try_execute(download_file, 5, {"url": du["data_link"], "destination": destination})

        # Skip data units of type video from being added to the db (they are added after the video processing stage)
        if label_row.data_type == DataType.VIDEO.value:
            return

        # Add non-video type of data to the db
        with PrismaConnection(project_file_structure) as conn:
            conn.dataunit.upsert(
                where={
                    "data_hash_frame": {  # state the values of the compound key
                        "data_hash": du["data_hash"],
                        "frame": int(du["data_sequence"]),
                    }
                },
                data={
                    "create": {
                        "data_hash": du["data_hash"],
                        "data_title": du["data_title"],
                        "frame": int(du["data_sequence"]),
                        "location": destination.resolve().as_posix(),
                        "lr_data_hash": label_row.data_hash,
                    },
                    "update": {
                        "data_title": du["data_title"],
                        "location": destination.resolve().as_posix(),
                    },
                },
            )


def download_label_row_and_data(
    label_hash: str, project: EncordProject, project_file_structure: ProjectFileStructure
) -> Optional[LabelRow]:

    label_row = download_label_row(label_hash, project, project_file_structure)
    download_data(label_row, project_file_structure)
    return label_row


def split_lr_videos(label_rows: List[LabelRow], project_file_structure: ProjectFileStructure) -> List[bool]:
    return collect_async(
        partial(split_lr_video, project_file_structure=project_file_structure),
        filter(lambda lr: lr.data_type == "video", label_rows),
        desc="Splitting videos into frames",
    )


def split_lr_video(label_row: LabelRow, project_file_structure: ProjectFileStructure) -> bool:
    """
    Take a label row, if it is a video, split the underlying video file into frames.
    :param label_row: The label row to consider splitting.
    :param project_file_structure: The directory of the project.
    :return: Whether the label row had a video to split.
    """
    if label_row.data_type == "video":
        lr_structure = project_file_structure.label_row_structure(label_row.label_hash)
        data_hash = list(label_row.data_units.keys())[0]
        du = label_row.data_units[data_hash]
        suffix = f".{du['data_type'].split('/')[1]}"
        video_path = (lr_structure.images_dir / du["data_hash"]).with_suffix(suffix)
        sliced_frames, _ = slice_video_into_frames(video_path)

        # 'create_many' behaviour is not available for SQLite in prisma, so batch creation is the way to go
        with PrismaConnection(project_file_structure) as conn:
            with conn.batch_() as batcher:
                sliced_frames[-1] = video_path  # To include a reference to the video location in the DataUnit table
                for frame_num, frame_path in sliced_frames.items():
                    batcher.dataunit.create(
                        data={
                            "data_hash": du["data_hash"],
                            "data_title": du["data_title"],
                            "frame": frame_num,
                            "location": frame_path.resolve().as_posix(),
                            "lr_data_hash": label_row.data_hash,
                        }
                    )
                batcher.commit()
        return True
    return False
