from __future__ import annotations

import itertools
import json
import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

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
        self.project_dir: Path = project_dir
        self.project_file_structure = ProjectFileStructure(project_dir)
        self.project_meta = fetch_project_meta(self.project_file_structure.data)
        self.project_hash: str = ""
        self.ontology: OntologyStructure = OntologyStructure.from_dict(dict(objects=[], classifications=[]))
        self.label_row_meta: Dict[str, LabelRowMetadata] = {}
        self.label_rows: Dict[str, LabelRow] = {}
        self.image_paths: Dict[str, List[Path]] = {}

    def load(self, subset_size: Optional[int] = None) -> Project:
        """
        Initialize a project from local storage.

        :param subset_size: The number of label rows to fetch. If None then all label rows are fetched.
        :return: The updated project instance.
        """
        if self.is_loaded:
            return self

        if not self.project_dir.exists():
            raise FileNotFoundError(f"`{self.project_dir}` does not exist")
        if not self.project_dir.is_dir():
            raise NotADirectoryError(f"`{self.project_dir}` does not point to a directory")

        self.project_meta = fetch_project_meta(self.project_dir)
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

        self.project_dir.mkdir(parents=True, exist_ok=True)

        # todo enforce clean up when we are sure it won't impact performance in other sections (like PredictionWriter)
        # also don't forget to add `from shutil import rmtree` at the top (pylint tags it as unused right now)
        # clean project_dir content
        # for path in self.project_dir.iterdir():
        #     if path.is_file():
        #         path.unlink()
        #     elif path.is_dir():
        #         rmtree(path)

        self.__save_project_meta(encord_project)
        self.__save_ontology(encord_project)
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
                    self.label_row_meta,
                    self.label_rows,
                ],
            )
        )

    def __save_project_meta(self, encord_project: EncordProject):
        project_meta_file_path = self.project_file_structure.project_meta
        self.project_meta.update(
            {
                "project_title": encord_project.title,
                "project_description": encord_project.description,
                "project_hash": encord_project.project_hash,
            }
        )
        project_meta_file_path.write_text(yaml.safe_dump(self.project_meta), encoding="utf-8")

    def __save_ontology(self, encord_project: EncordProject):
        ontology_file_path = self.project_file_structure.ontology
        ontology_file_path.write_text(json.dumps(encord_project.ontology, indent=2), encoding="utf-8")

    def __load_ontology(self):
        ontology_file_path = self.project_file_structure.ontology
        if not ontology_file_path.exists():
            raise FileNotFoundError(f"Expected file `ontology.json` at {ontology_file_path.parent}")
        self.ontology = OntologyStructure.from_dict(json.loads(ontology_file_path.read_text(encoding="utf-8")))

    def __save_label_row_meta(self, encord_project: EncordProject):
        label_row_meta = {lr["label_hash"]: lr for lr in encord_project.label_rows if lr["label_hash"] is not None}
        label_row_meta_file_path = self.project_file_structure.label_row_meta
        label_row_meta_file_path.write_text(json.dumps(label_row_meta, indent=2), encoding="utf-8")

    def __load_label_row_meta(self, subset_size: Optional[int]):
        label_row_meta_file_path = self.project_file_structure.label_row_meta
        if not label_row_meta_file_path.exists():
            raise FileNotFoundError(f"Expected file `label_row_meta.json` at {label_row_meta_file_path.parent}")
        self.label_row_meta = {
            lr_hash: LabelRowMetadata.from_dict(lr_meta)
            for lr_hash, lr_meta in itertools.islice(
                json.loads(label_row_meta_file_path.read_text(encoding="utf-8")).items(), subset_size
            )
        }

    def __download_and_save_label_rows(self, encord_project: EncordProject):
        label_rows = download_all_label_rows(encord_project, cache_dir=self.project_dir)
        download_all_images(label_rows, cache_dir=self.project_dir)

    def __load_label_rows(self):
        self.label_rows = {}
        self.image_paths = {}
        for lr_hash in self.label_row_meta.keys():
            lr_file_path = self.project_file_structure.label_row_structure(lr_hash).label_row_file
            lr_images_dir = self.project_file_structure.data / lr_hash / "images"
            if not lr_file_path.is_file() or not lr_images_dir.is_dir():
                logger.warning(
                    f"Skipping label row <blue>`{lr_hash}`</blue> as no stored content was found for the label row."
                )
                continue
            self.label_rows[lr_hash] = LabelRow(json.loads(lr_file_path.read_text(encoding="utf-8")))
            self.image_paths[lr_hash] = list(lr_images_dir.iterdir())


def get_label_row(lr, client, cache_dir, refresh=False) -> Optional[LabelRow]:
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    if not lr["label_hash"]:
        return None

    cache_pth = cache_dir / "data" / lr["label_hash"] / "label_row.json"

    if not refresh and cache_pth.is_file():
        try:
            with cache_pth.open("r") as f:
                return LabelRow(json.load(f))
        except json.decoder.JSONDecodeError:
            pass

    try:
        lr = client.get_label_row(lr["label_hash"])
    except encord.exceptions.UnknownException:
        logger.warning(
            f"Failed to download label row with label_hash <blue>`{lr['label_hash'][:8]}`</blue> and data_title <blue>`{lr['data_title']}`</blue>"
        )
        return None

    cache_pth.parent.mkdir(parents=True, exist_ok=True)
    with cache_pth.open("w") as f:
        json.dump(lr, f, indent=2)

    return lr


def download_all_label_rows(client, subset_size: Optional[int] = None, **kwargs) -> Dict[str, LabelRow]:
    label_rows = list(itertools.islice(filter(lambda x: x["label_hash"], client.label_rows), subset_size))

    return collect_async(
        partial(get_label_row, client=client, **kwargs),
        label_rows,
        lambda lr: lr["label_hash"],
        desc="Collecting label rows from Encord SDK.",
    )


def download_images_from_data_unit(lr, cache_dir, **kwargs) -> Optional[List[Path]]:
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    label_hash = lr.label_hash

    if label_hash is None:
        return None

    label_pth = cache_dir / "data" / label_hash
    label_pth.mkdir(parents=True, exist_ok=True)

    lr_path = label_pth / "label_row.json"
    if not lr_path.exists():
        with (label_pth / "label_row.json").open("w") as f:
            json.dump(lr, f, indent=2)

    frame_pth = label_pth / "images"

    frame_pth.mkdir(parents=True, exist_ok=True)
    frame_pths: List[Path] = []
    data_units = sorted(lr.data_units.values(), key=lambda du: int(du["data_sequence"]))
    for du in data_units:
        suffix = f".{du['data_type'].split('/')[1]}"
        out_pth = (frame_pth / du["data_hash"]).with_suffix(suffix)
        out_pth = download_file(du["data_link"], out_pth)
        frame_pths.append(out_pth)

    if lr.data_type == "video":
        video_path = frame_pths[0]
        frame_pths.clear()
        for out_pth in slice_video_into_frames(video_path)[0].values():
            frame_pths.append(out_pth)

    return frame_pths


def download_all_images(label_rows, cache_dir: Union[str, Path], **kwargs) -> Dict[str, List[Path]]:
    return collect_async(
        partial(download_images_from_data_unit, cache_dir=cache_dir, **kwargs),
        label_rows.values(),
        lambda lr: lr.label_hash,
        desc="Collecting frames from label rows.",
    )
