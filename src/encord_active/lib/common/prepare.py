import itertools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import encord.exceptions
import requests
from encord import Project
from encord.orm.label_row import LabelRow
from encord.project import LabelRowMetadata
from tqdm import tqdm

from encord_active.lib.common.utils import fetch_project_meta, slice_video_into_frames

logger = logging.getLogger(__name__)
encord_logger = logging.getLogger("encord")
encord_logger.setLevel(logging.ERROR)


@dataclass
class EAProject:
    label_rows: Dict[str, LabelRow]
    label_row_meta: Dict[str, LabelRowMetadata]
    image_paths: Dict[str, List[Path]]
    project_meta: Dict[str, str]
    ontology: Dict[str, dict]
    project_hash: str = field(init=False)

    def __post_init__(self):
        self.project_hash = self.project_meta["project_hash"]


def collect_async(fn, job_args, key_fn, max_workers=min(10, (os.cpu_count() or 1) + 4), **kwargs):
    """
    Distribute work across multiple workers. Good for, e.g., downloading data.
    Will return results in dictionary.
    :param fn: The function to be applied
    :param job_args: Arguments to `fn`.
    :param key_fn: Function to determine dictionary key for the result (given the same input as `fn`).
    :param max_workers: Number of workers to distribute work over.
    :param kwargs: Arguments passed on to tqdm.
    :return: Dictionary {key_fn(*job_args): fn(*job_args)}
    """
    job_args = list(job_args)
    if not isinstance(job_args[0], tuple):
        job_args = [(j,) for j in job_args]

    results = {}
    with tqdm(total=len(job_args), **kwargs) as pbar:
        with Executor(max_workers=max_workers) as exe:
            jobs = {exe.submit(fn, *args): key_fn(*args) for args in job_args}
            for job in as_completed(jobs):
                key = jobs[job]

                result = job.result()
                if result:
                    results[key] = result

                pbar.update(1)
    return results


def download_file(
    url: str,
    destination: Path,
    byte_size=1024,
):
    if destination.is_file():
        return destination

    r = requests.get(url, stream=True)
    with destination.open("wb") as f:
        for chunk in r.iter_content(chunk_size=byte_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return destination


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


def read_project_data_from_cache(cache_dir: Path, subset_size: Optional[int]) -> EAProject:
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

    project_meta = fetch_project_meta(cache_dir)
    label_row_meta = json.loads((cache_dir / "label_row_meta.json").read_text())
    ontology = json.loads((cache_dir / "ontology.json").read_text())
    image_paths = download_all_images(label_rows, cache_dir)

    return EAProject(
        label_rows=label_rows,
        label_row_meta=label_row_meta,
        image_paths=image_paths,
        ontology=ontology,
        project_meta=project_meta,
    )


def prepare_data(
    cache_dir: Path, subset_size: Optional[int] = None, project: Optional[Project] = None, **kwargs
) -> EAProject:
    """
    Fetches data either from cache or from the Encord project if it's available.

    :param cache_dir: The root directory of the project.
    :param subset_size: The number of label rows to fetch. None means all.
    :param project: An encord project for fetching data. If not provided, will only use cached files.

    :returns: all the raw project data.
    """
    if project is None:
        return read_project_data_from_cache(cache_dir, subset_size=subset_size)

    label_rows = download_all_label_rows(project, subset_size=subset_size, cache_dir=cache_dir, **kwargs)
    image_paths = download_all_images(label_rows, cache_dir=cache_dir, **kwargs)

    project_meta = fetch_project_meta(cache_dir)
    # Cache ontology object
    ontology_file = cache_dir / "ontology.json"
    if not ontology_file.exists():
        with ontology_file.open("w") as f:
            json.dump(project.ontology, f)

    label_row_meta = {lr["label_hash"]: lr for lr in project.label_rows if lr["label_hash"] is not None}
    # Cache label rows metadata
    label_row_metadata_file = cache_dir / "label_row_meta.json"
    if not label_row_metadata_file.exists():
        with label_row_metadata_file.open("w") as f:
            json.dump(label_row_meta, f)

    return EAProject(
        label_row_meta=label_row_meta,
        label_rows=label_rows,
        project_meta=project_meta,
        image_paths=image_paths,
        ontology=project.ontology,
    )
