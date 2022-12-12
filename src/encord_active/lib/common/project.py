from __future__ import annotations

import dataclasses
import itertools
import json
import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import encord.exceptions
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


@dataclasses.dataclass
class Project:
    image_paths: Dict[str, List[Path]]
    label_rows: Dict[str, LabelRow]
    label_row_meta: Dict[str, LabelRowMetadata]
    ontology: OntologyStructure
    project_hash: str
    project_meta: Dict[str, str]

    @staticmethod
    def read(
        cache_dir: Path,
        subset_size: Optional[int] = None,
        encord_project: Optional[EncordProject] = None,
        **kwargs,
    ) -> Project:
        """
        Fetches data either from cache or from the Encord project if it's available.

        :param cache_dir: The root directory of the project.
        :param subset_size: The number of label rows to fetch. None means all.
        :param encord_project: An encord project for fetching data. If not provided, will only use cached files.
        """
        if encord_project is None:
            label_rows = dict(
                itertools.islice(
                    (
                        (p.name, LabelRow(json.loads((p / "label_row.json").read_text())))
                        for p in (cache_dir / "data").iterdir()
                        if (p / "label_row.json").is_file()
                    ),
                    subset_size,
                )
            )
            image_paths = download_all_images(label_rows, cache_dir)

            label_row_meta = json.loads((cache_dir / "label_row_meta.json").read_text())
            ontology = json.loads((cache_dir / "ontology.json").read_text())
            project_meta = fetch_project_meta(cache_dir)
            project_hash = project_meta["project_hash"]
        else:
            label_rows = download_all_label_rows(  # todo check this
                encord_project, subset_size=subset_size, cache_dir=cache_dir, **kwargs
            )
            image_paths: Dict[str, List[Path]] = download_all_images(
                label_rows, cache_dir=cache_dir, **kwargs
            )  # todo check this

            label_row_meta: Dict[str, LabelRowMetadata] = {
                lr["label_hash"]: LabelRowMetadata.from_dict(lr)
                for lr in encord_project.label_rows
                if lr["label_hash"] is not None
            }
            ontology = OntologyStructure.from_dict(encord_project.ontology)
            project_meta = fetch_project_meta(cache_dir)  # todo check this
            project_hash = project_meta["project_hash"]

        return Project(
            image_paths=image_paths,
            label_rows=label_rows,
            label_row_meta=label_row_meta,
            ontology=ontology,
            project_hash=project_hash,
            project_meta=project_meta,
        )

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
