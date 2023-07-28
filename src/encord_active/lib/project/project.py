from __future__ import annotations

import itertools
import json
import logging
import tempfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    import prisma
    from prisma.types import DataUnitUpsertInput

import yaml
from encord import Project as EncordProject
from encord.constants.enums import DataType
from encord.objects.ontology_structure import OntologyStructure
from encord.orm.label_row import LabelRow
from encord.project import LabelRowMetadata
from loguru import logger
from PIL import Image

from encord_active.cli.config import app_config
from encord_active.lib.common.data_utils import (
    collect_async,
    count_frames,
    download_file,
    extract_frames,
    file_path_to_url,
    get_frames_per_second,
    try_execute,
)
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.db.prisma_init import ensure_prisma_db
from encord_active.lib.encord.utils import get_client, handle_enum_and_datetime
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
                project_description="",
                project_hash="",
                project_title="",
                ssh_key_path="",
                has_remote=False,
                data_version=0,
                store_data_locally=False,
            )
        self.project_hash: str = ""
        self.ontology: OntologyStructure = OntologyStructure.from_dict(dict(objects=[], classifications=[]))
        self.label_row_metas: Dict[str, LabelRowMetadata] = {}
        self.label_rows: Dict[str, LabelRow] = {}

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
        new_lr_structure = json.dumps(label_row)
        with PrismaConnection(self.file_structure) as conn:
            conn.labelrow.update(
                data={"label_row_json": new_lr_structure},
                where={"label_hash": label_row["label_hash"]},
            )

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
            with PrismaConnection(project_file_structure) as conn:
                with conn.batch_() as batch:
                    collect_async(
                        partial(download_label_row, project=project, batch=batch),
                        label_rows_to_update,
                        desc="Updating label rows",
                    )
                    batch.commit()
        else:
            logger.info("No existent data needs to be updated.")

        # Download new project data
        if len(label_rows_to_download) > 0:
            operation_description = "Collecting the new data"
            store_data_locally: bool = project_file_structure.load_project_meta().get("store_data_locally", False)
            if not store_data_locally:
                operation_description = "Collecting information on the new data"

            with PrismaConnection(project_file_structure) as conn:
                with conn.batch_() as batch:
                    downloaded_label_rows = collect_async(
                        partial(
                            download_label_row_and_data,
                            project=project,
                            project_file_structure=project_file_structure,
                            batch=batch,
                        ),
                        label_rows_to_download,
                        desc=operation_description,
                    )
                    batch.commit()
        else:
            downloaded_label_rows = []
            logger.info("No new data to be downloaded.")

        return downloaded_label_rows

    def __load_label_rows(self):
        self.label_rows = {}
        with PrismaConnection(self.file_structure) as conn:
            labels = conn.labelrow.find_many(
                include={
                    "data_units": True,
                }
            )
            for label in labels:
                self.label_rows[label.label_hash] = LabelRow(json.loads(label.label_row_json))

    def __populate_label_row_metadata_defaults(self, lr_dict: dict):
        return {
            "data_link": "null",
            "dataset_title": "",
            "is_shadow_data": False,
            "number_of_frames": 1,
            **lr_dict,
        }


def download_label_row(
    label_hash: str,
    project: EncordProject,
    batch: "prisma.Batch",
) -> LabelRow:
    label_row = try_execute(partial(project.get_label_row, get_signed_url=True), 5, {"uid": label_hash})
    label_row_json = json.dumps(label_row)
    batch.labelrow.upsert(
        where={"data_hash": label_row.data_hash},
        data={
            "create": {
                "label_hash": label_row.label_hash,
                "data_hash": label_row.data_hash,
                "data_title": label_row.data_title,
                "data_type": label_row.data_type,
                "created_at": label_row.created_at,
                "last_edited_at": label_row.last_edited_at,
                "label_row_json": label_row_json,
            },
            "update": {
                "label_hash": label_row.label_hash,
                "data_title": label_row.data_title,
                "created_at": label_row.created_at,  # don't update this field if it's set in unannotated data
                "last_edited_at": label_row.last_edited_at,
                "label_row_json": label_row_json,
            },
        },
    )
    return label_row


def download_data(
    label_row: LabelRow,
    project_file_structure: ProjectFileStructure,
    batch: "prisma.Batch",
):
    store_data_locally: bool = project_file_structure.load_project_meta().get("store_data_locally", False)
    if store_data_locally:
        project_file_structure.local_data_store.mkdir(exist_ok=True)
    data_units = sorted(label_row.data_units.values(), key=lambda _du: int(_du["data_sequence"]))

    # Skip video frames from being added to the db (they are added after the video processing stage)
    if label_row.data_type == DataType.VIDEO.value:
        return

    for du in data_units:
        data_hash = du["data_hash"]
        frame = int(du["data_sequence"])
        width = du["width"]
        height = du["height"]

        # State what data is going to be created / updated in the db upsert
        query_data_input: DataUnitUpsertInput = {
            "create": {
                "data_hash": data_hash,
                "data_title": du["data_title"],
                "frame": frame,
                "lr_data_hash": label_row.data_hash,
                "fps": 0,
                "width": width,
                "height": height,
            },
            "update": {
                "data_title": du["data_title"],
            },
        }

        if store_data_locally:
            signed_url = du["data_link"]
            local_path = project_file_structure.local_data_store / data_hash
            title_suffix = Path(du["data_title"]).suffix
            local_path = local_path.with_suffix(title_suffix)
            download_file(
                signed_url,
                project_file_structure.project_dir,
                local_path,
                cache=False,  # Disable cache symlink tricks
            )
            file_url = file_path_to_url(local_path, project_file_structure.project_dir)
            query_data_input["create"]["data_uri"] = query_data_input["update"]["data_uri"] = file_url
        else:
            # The online version doesn't require any additional content, except for the image url,
            # which is not a permalink so best not to add it.
            pass

        batch.dataunit.upsert(
            # State the compound key representing the data unit in the db
            where={"data_hash_frame": {"data_hash": data_hash, "frame": frame}},
            data=query_data_input,
        )


def download_label_row_and_data(
    label_hash: str,
    project: EncordProject,
    project_file_structure: ProjectFileStructure,
    batch: "prisma.Batch",
) -> Optional[LabelRow]:
    label_row = download_label_row(label_hash, project, batch)
    download_data(label_row, project_file_structure, batch)
    return label_row


def split_lr_videos(label_rows: List[LabelRow], project_file_structure: ProjectFileStructure) -> List[bool]:
    return collect_async(
        partial(split_lr_video, project_file_structure=project_file_structure),
        filter(lambda lr: lr.data_type == "video", label_rows),
        desc="Splitting videos into frames",
    )


def split_lr_video(label_row: LabelRow, project_file_structure: ProjectFileStructure) -> bool:
    store_data_locally: bool = project_file_structure.load_project_meta().get("store_data_locally", False)
    if store_data_locally:
        project_file_structure.local_data_store.mkdir(exist_ok=True)
    """
    Take a label row, if it is a video, split the underlying video file into frames.
    :param label_row: The label row to consider splitting.
    :param project_file_structure: The directory of the project.
    :return: Whether the label row had a video to split.
    """
    if label_row.data_type == "video":
        data_hash = list(label_row.data_units.keys())[0]
        du = label_row.data_units[data_hash]
        with tempfile.TemporaryDirectory() as video_dir:
            if store_data_locally:
                video_path = (project_file_structure.local_data_store / data_hash).with_suffix(
                    Path(du["data_title"]).suffix
                )
                data_uri = file_path_to_url(video_path, project_dir=project_file_structure.project_dir)
                download_file(
                    du["data_link"],
                    project_file_structure.project_dir,
                    video_path,
                    cache=False,  # Disable cache symlink tricks
                )
            else:
                video_path = Path(video_dir) / du["data_title"]
                data_uri = None
                download_file(du["data_link"], project_dir=project_file_structure.project_dir, destination=video_path)
                project_file_structure.cached_signed_urls[du["data_hash"]] = du["data_link"]
            num_frames = count_frames(video_path)
            frames_per_second = get_frames_per_second(video_path)
            video_images = Path(video_dir) / "images"
            extract_frames(video_path, video_images, data_hash)
            image_path = next(video_images.iterdir())
            image = Image.open(image_path)

        # 'create_many' behaviour is not available for SQLite in prisma, so batch creation is the way to go
        with PrismaConnection(project_file_structure) as conn:
            with conn.batch_() as batcher:
                for frame_num in range(num_frames):
                    batcher.dataunit.create(
                        data={
                            "data_hash": du["data_hash"],
                            "data_title": du["data_title"],
                            "frame": frame_num,
                            "lr_data_hash": label_row.data_hash,
                            "width": image.width,
                            "height": image.height,
                            "fps": frames_per_second,
                            "data_uri": data_uri,
                        }
                    )
                batcher.commit()
        return True
    return False
