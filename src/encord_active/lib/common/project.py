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

from encord_active.lib.common.utils import (
    collect_async,
    download_file,
    fetch_project_meta,
    slice_video_into_frames,
)

logger = logging.getLogger(__name__)
encord_logger = logging.getLogger("encord")
encord_logger.setLevel(logging.ERROR)


class Project:
    def __init__(self, project_dir: Path, subset_size: Optional[int] = None):
        """
        Fetch project's data from local storage.

        :param project_dir: The root directory of the project.
        :param subset_size: The number of label rows to fetch. If None then all label rows are fetched.
        """
        if not project_dir.exists():
            raise FileNotFoundError("`project_dir` must point to an existing directory")
        if not project_dir.is_dir():
            raise NotADirectoryError("`project_dir` must point to an existing directory")
        self.project_dir: Path = project_dir

        # read project's metadata and hash
        self.project_meta: Dict[str, str] = fetch_project_meta(project_dir)
        self.project_hash: str = self.project_meta["project_hash"]

        # read project ontology
        ontology_file_path = project_dir / "ontology.json"
        if not ontology_file_path.exists():
            raise FileNotFoundError(f"Expected file `ontology.json` at {project_dir}")
        self.ontology: OntologyStructure = OntologyStructure.from_dict(
            json.loads((project_dir / "ontology.json").read_text())
        )

        # read label rows' metadata
        label_row_meta_file_path = project_dir / "label_row_meta.json"
        if not label_row_meta_file_path.exists():
            raise FileNotFoundError(f"Expected file `label_row_meta.json` at {project_dir}")
        self.label_row_meta: Dict[str, LabelRowMetadata] = {}
        for lr_hash, lr_meta in itertools.islice(json.loads(label_row_meta_file_path.read_text()).items(), subset_size):
            self.label_row_meta[lr_hash] = LabelRowMetadata.from_dict(lr_meta)

        # read label rows and their images
        self.label_rows: Dict[str, LabelRow] = {}
        self.image_paths: Dict[str, List[Path]] = {}
        for lr_hash in self.label_row_meta.keys():
            lr_file_path = project_dir / "data" / lr_hash / "label_row.json"
            lr_images_dir = project_dir / "data" / lr_hash / "images"
            if not lr_file_path.exists():  # todo log this issue
                continue
            if not lr_images_dir.exists() or not lr_images_dir.is_dir():  # todo log this issue
                continue
            self.label_rows[lr_hash] = LabelRow(json.loads(lr_file_path.read_text()))
            self.image_paths[lr_hash] = list(lr_images_dir.iterdir())

    @classmethod
    def from_encord_project(cls, project_dir: Path, encord_project: EncordProject):
        """
        Construct a Project from Encord platform.

        :param project_dir: The root directory of the project.
        :param encord_project: Encord project from where the data is fetched.
        :return:
        """
        if not project_dir.exists():
            raise FileNotFoundError("`project_dir` must point to an existing directory")
        if not project_dir.is_dir():
            raise NotADirectoryError("`project_dir` must point to an existing directory")

        # todo enforce clean up when we are sure it won't impact performance in other sections (like PredictionWriter)
        # also don't forget to add `from shutil import rmtree` at the top (pylint tags it as unused right now)
        # clean project_dir content
        # for path in project_dir.iterdir():
        #     if path.is_file():
        #         path.unlink()
        #     elif path.is_dir():
        #         rmtree(path)

        # store project's metadata
        project_meta = {
            "project_title": encord_project.title,
            "project_description": encord_project.description,
            "project_hash": encord_project.project_hash,
        }
        project_meta_file_path = project_dir / "project_meta.yaml"
        project_meta_file_path.write_text(yaml.dump(project_meta))  # , encoding="utf-8")

        # store project's ontology
        ontology_file_path = project_dir / "ontology.json"
        ontology_file_path.write_text(json.dumps(encord_project.ontology, indent=2))  # , encoding="utf-8")

        # store label rows' metadata
        label_row_meta = {lr["label_hash"]: lr for lr in encord_project.label_rows if lr["label_hash"] is not None}
        label_row_meta_file_path = project_dir / "label_row_meta.json"
        label_row_meta_file_path.write_text(json.dumps(label_row_meta, indent=2))  # , encoding="utf-8")

        # store label rows and their images
        label_rows = download_all_label_rows(encord_project, cache_dir=project_dir)  # todo no need to output all data
        image_paths = download_all_images(label_rows, cache_dir=project_dir)  # todo no need to output all data

        return Project(project_dir)

    # unused method (prototype)
    def save(self, cache_dir: Path) -> None:
        ontology_file = cache_dir / "ontology.json"
        if not ontology_file.exists():
            with ontology_file.open("w") as f:
                json.dump(self.ontology.to_dict(), f)

        label_row_meta_file = cache_dir / "label_row_meta.json"
        if not label_row_meta_file.exists():
            with label_row_meta_file.open("w") as f:
                # TODO uncomment next line when LabelRowMetadata's to_dict() method get added in Encord SDK
                # label_row_meta_dict = {k: v.to_dict() for k, v in self.label_row_meta}
                label_row_meta_dict = {k: labelrowmetadata_to_dict(v) for k, v in self.label_row_meta.items()}
                json.dump(label_row_meta_dict, f)


# Temporary method. Should last while the related PR in Encord SDK is not in prod
def labelrowmetadata_to_dict(self) -> dict:
    """
    Returns:
        The dict equivalent of LabelRowMetadata.
    """
    return dict(
        label_hash=self.label_hash,
        label_status=self.label_status.value,
        data_hash=self.data_hash,
        dataset_hash=self.dataset_hash,
        data_title=self.data_title,
        data_type=self.data_type,
        is_shadow_data=self.is_shadow_data,
        annotation_task_status=self.annotation_task_status.value,
    )


def get_label_row(lr, client, cache_dir, refresh=False) -> Optional[LabelRow]:
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    if not lr["label_hash"]:
        return None

    cache_pth = cache_dir / "data" / lr["label_hash"] / "label_row.json"

    if not refresh and cache_pth.is_file():
        # Load cached version
        try:
            with cache_pth.open("r") as f:
                return LabelRow(json.load(f))
        except json.decoder.JSONDecodeError:
            pass

    try:
        lr = client.get_label_row(lr["label_hash"])
    except encord.exceptions.UnknownException as e:
        logger.warning(
            f"Failed to download label row with label_hash {lr['label_hash'][:8]}... and data_title {lr['data_title']}"
        )
        return None

    # Cache label row.
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

    # if label row's data type is video then extract frames
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
