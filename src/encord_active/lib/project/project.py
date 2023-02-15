from __future__ import annotations

import itertools
import json
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import encord.exceptions
import yaml
from encord import Project as EncordProject
from encord.objects.ontology_structure import OntologyStructure
from encord.orm.label_row import LabelRow
from encord.project import LabelRowMetadata
from loguru import logger

from encord_active.lib.common.utils import (
    collect_async,
    download_file,
    fetch_project_meta,
    slice_video_into_frames,
)
from encord_active.lib.project.project_file_structure import ProjectFileStructure

logger = logger.opt(colors=True)
encord_logger = logging.getLogger("encord")
encord_logger.setLevel(logging.ERROR)


class Project:
    def __init__(self, project_dir: Path):
        self.file_structure = ProjectFileStructure(project_dir)
        self.project_meta = fetch_project_meta(self.file_structure.project_dir)
        self.project_hash: str = ""
        self.ontology: OntologyStructure = OntologyStructure.from_dict(dict(objects=[], classifications=[]))
        self.label_row_metas: Dict[str, LabelRowMetadata] = {}
        self.label_rows: Dict[str, LabelRow] = {}
        self.image_paths: Dict[str, Dict[str, Path]] = {}

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
        self.__save_label_row_meta(encord_project)
        self.__download_and_save_label_rows(encord_project)

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
        label_row_meta = {lr["label_hash"]: lr for lr in encord_project.label_rows if lr["label_hash"] is not None}
        label_row_meta_file_path = self.file_structure.label_row_meta
        label_row_meta_file_path.write_text(json.dumps(label_row_meta, indent=2), encoding="utf-8")

    def __load_label_row_meta(self, subset_size: Optional[int]):
        label_row_meta_file_path = self.file_structure.label_row_meta
        if not label_row_meta_file_path.exists():
            raise FileNotFoundError(f"Expected file `label_row_meta.json` at {label_row_meta_file_path.parent}")
        self.label_row_metas = {
            lr_hash: LabelRowMetadata.from_dict(lr_meta)
            for lr_hash, lr_meta in itertools.islice(
                json.loads(label_row_meta_file_path.read_text(encoding="utf-8")).items(), subset_size
            )
        }

    def save_label_row(self, label_row: LabelRow):
        lr_structure = self.file_structure.label_row_structure(label_row["label_hash"])
        lr_structure.label_row_file.write_text(json.dumps(label_row, indent=2), encoding="utf-8")

    def __download_and_save_label_rows(self, encord_project: EncordProject):
        label_rows = download_label_rows(encord_project, self.file_structure)
        results = download_all_images(label_rows, self.file_structure)

        lr_with_failed_images = {lr_hash for lr_hash, success in results.items() if success is False}

        if not lr_with_failed_images:
            logger.info("Successfully downloaded all label row datas")
            return

        # NOTE: most likely the singed url have expired, we should refresh the label row and try again.
        logger.warning(
            f"Failed downloading datas for some the following label rows: {lr_with_failed_images}. \nRetrying..."
        )
        label_rows = download_label_rows(
            self.encord_project,
            self.file_structure,
            filter_fn=lambda lr: lr["label_hash"] in lr_with_failed_images,
            refresh=True,
        )
        results = download_all_images(label_rows, self.file_structure)
        lr_with_failed_images = {lr_hash for lr_hash, success in results.items() if success is False}

        if not lr_with_failed_images:
            logger.info("Successfully downloaded all label row datas")
            return

        logger.error("Retry didn't help. Skipping...")

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


def get_label_row(
    lr, project: EncordProject, project_file_structure: ProjectFileStructure, refresh=False
) -> Optional[LabelRow]:
    if not lr["label_hash"]:
        return None

    lr_structure = project_file_structure.label_row_structure(lr["label_hash"])
    lr_structure.path.mkdir(parents=True, exist_ok=True)

    if not refresh and lr_structure.label_row_file.is_file():
        try:
            return LabelRow(json.loads(lr_structure.label_row_file.read_text(encoding="utf-8")))
        except json.decoder.JSONDecodeError:
            pass

    try:
        lr = project.get_label_row(lr["label_hash"])
    except encord.exceptions.UnknownException:
        logger.warning(
            f"Failed to download label row with label_hash <blue>`{lr['label_hash']}`</blue> and data_title <blue>`{lr['data_title']}`</blue>"
        )
        return None

    lr_structure.label_row_file.write_text(json.dumps(lr, indent=2), encoding="utf-8")
    return lr


def download_label_rows(
    project: EncordProject,
    project_file_structure: ProjectFileStructure,
    filter_fn: Optional[Callable[..., bool]] = lambda x: x["label_hash"] is not None,
    subset_size: Optional[int] = None,
    **kwargs,
) -> Dict[str, LabelRow]:
    label_rows = list(itertools.islice(filter(filter_fn, project.label_rows), subset_size))

    return collect_async(
        partial(get_label_row, project=project, project_file_structure=project_file_structure, **kwargs),
        label_rows,
        lambda lr: lr["label_hash"],
        desc="Collecting label rows from Encord SDK",
    )


def download_images_from_data_unit(lr: LabelRow, project_file_structure: ProjectFileStructure) -> Optional[bool]:
    label_hash = lr.label_hash

    if label_hash is None:
        return None

    lr_structure = project_file_structure.label_row_structure(label_hash)
    lr_structure.path.mkdir(parents=True, exist_ok=True)

    if not lr_structure.label_row_file.exists():
        lr_structure.label_row_file.write_text(json.dumps(lr, indent=2), encoding="utf-8")

    lr_structure.images_dir.mkdir(parents=True, exist_ok=True)
    frame_pths: List[Path] = []
    data_units = sorted(lr.data_units.values(), key=lambda du: int(du["data_sequence"]))
    for du in data_units:
        suffix = f".{du['data_type'].split('/')[1]}"
        out_pth = (lr_structure.images_dir / du["data_hash"]).with_suffix(suffix)
        try:
            out_pth = download_file(du["data_link"], out_pth)
            frame_pths.append(out_pth)
        except:
            return False

    if lr.data_type == "video":
        video_path = frame_pths[0]
        frame_pths.clear()
        for out_pth in slice_video_into_frames(video_path)[0].values():
            frame_pths.append(out_pth)

    return True


def download_all_images(label_rows, project_file_structure: ProjectFileStructure) -> Dict[str, Optional[bool]]:
    return collect_async(
        partial(download_images_from_data_unit, project_file_structure=project_file_structure),
        label_rows.values(),
        lambda lr: lr.label_hash,
        desc="Collecting frames from label rows",
    )
