import os
import shutil
from pathlib import Path
from typing import Callable, Optional, TypedDict

import requests
import rich
import typer
from encord_active_components.components.projects_page import ProjectStats
from rich.markup import escape
from tqdm.auto import tqdm


class PrebuiltProject(TypedDict):
    url: str
    hash: str
    name: str
    image_path: Path
    stats: ProjectStats


IMAGES_PATH = (Path(__file__).parent).resolve() / "images"

# GCP bucket links will be added here
PREBUILT_PROJECTS: dict[str, PrebuiltProject] = {
    "[open-source][validation]-coco-2017-dataset": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Bvalidation%5D-coco-2017-dataset.zip",
        hash="f2140a72-c644-4c31-be66-3ef80b3718e5",
        name="[open-source][validation]-coco-2017-dataset",
        image_path=(IMAGES_PATH / "coco.jpeg"),
        stats=ProjectStats(dataUnits=4952, labels=41420, classes=80),
    ),
    "[open-source][test]-limuc-ulcerative-colitis-classification": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btest%5D-limuc-ulcerative-colitis-classification.zip",
        hash="aa2b21bd-6f2e-48fc-8f4f-4ba4d9b7bd67",
        name="[open-source][test]-limuc-ulcerative-colitis-classification",
        image_path=(IMAGES_PATH / "limuc.jpeg"),
        stats=ProjectStats(dataUnits=1686, labels=1686, classes=4),
    ),
    "[open-source]-covid-19-segmentations": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-covid-19-segmentations.zip",
        hash="d18819cb-2b75-4040-beb6-c63a901e6c84",
        name="[open-source]-covid-19-segmentations",
        image_path=(IMAGES_PATH / "covid_segmentations.jpeg"),
        stats=ProjectStats(dataUnits=100, labels=588, classes=13),
    ),
    "[open-source][validation]-bdd-dataset": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Bvalidation%5D-bdd-dataset.zip",
        hash="b37a48e0-6462-472d-baaa-2fcaf5ab9521",
        name="[open-source][validation]-bdd-dataset",
        image_path=(IMAGES_PATH / "bdd.jpeg"),
        stats=ProjectStats(dataUnits=981, labels=12983, classes=8),
    ),
    "[open-source]-TACO-Official": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-TACO-Official.zip",
        hash="dc1cf137-f1b9-4c2f-973d-32512c971955",
        name="[open-source]-TACO-Official",
        image_path=(IMAGES_PATH / "taco.jpeg"),
        stats=ProjectStats(dataUnits=1500, labels=5038, classes=59),
    ),
    "[open-source]-TACO-Unofficial": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-TACO-Unofficial.zip",
        hash="14a52852-55f6-46c9-850a-40e11540605f",
        name="[open-source]-TACO-Unofficial",
        image_path=(IMAGES_PATH / "taco_unofficial.jpeg"),
        stats=ProjectStats(dataUnits=3731, labels=8419, classes=60),
    ),
    "[open-source][train]-mnist-dataset": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btrain%5D-mnist-dataset.zip",
        hash="34413f3b-fed5-4a34-a279-b68a0c5fe325",
        name="[open-source][train]-mnist-dataset",
        image_path=IMAGES_PATH / "mnist.png",
        stats=ProjectStats(dataUnits=119000, labels=0, classes=0),
    ),
    "[open-source][test]-mnist-dataset": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btest%5D-mnist-dataset.zip",
        hash="5c96df58-beac-4e42-a74d-fce16622c5af",
        name="[open-source][test]-mnist-dataset",
        image_path=IMAGES_PATH / "mnist.png",
        stats=ProjectStats(dataUnits=70000, labels=0, classes=0),
    ),
    "rareplanes": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/rareplanes.zip",
        hash="1f4752d7-4a7a-4c0e-8b08-dd4b1c5a8bc6",
        name="rareplanes",
        image_path=(IMAGES_PATH / "rareplanes.jpeg"),
        stats=ProjectStats(dataUnits=2710, labels=6812, classes=7),
    ),
    "quickstart": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/quickstart.zip",
        hash="d6423838-f60e-41d9-b2ca-715aa2edef9c",
        name="quickstart",
        image_path=(IMAGES_PATH / "quickstart.jpeg"),
        stats=ProjectStats(dataUnits=199, labels=1617, classes=71),
    ),
}


def fetch_prebuilt_project_size(project_name: str) -> Optional[float]:
    """
    Returns the storage size (MB) of a prebuilt project stored in the cloud

    :param project_name: Prebuilt project name
    :return: Prebuilt project's storage size (MB). Can return None if the information is not available.
    """
    if project_name not in PREBUILT_PROJECTS.keys():
        return None
    url = PREBUILT_PROJECTS[project_name]["url"]
    with requests.head(url) as r:
        total_length = fetch_response_content_length(r)
        size = round(total_length / 1000000.0, 1) if total_length is not None else None
    return size


def fetch_response_content_length(r: requests.Response) -> Optional[int]:
    return int(r.headers["content-length"]) if "content-length" in r.headers.keys() else None


def fetch_prebuilt_project(
    project_name: str, out_dir: Path, *, unpack=True, progress_callback: Optional[Callable] = None
):
    url = PREBUILT_PROJECTS[project_name]["url"]
    output_file_name = "prebuilt_project.zip"
    output_file_path = out_dir / output_file_name
    rich.print(f"Output destination: {escape(out_dir.as_posix())}")
    out_dir.mkdir(exist_ok=True)

    if (out_dir / "project_meta.yaml").is_file():
        redownload = typer.confirm("Do you want to re-download the project?")
        if not redownload:
            return out_dir

    r = requests.get(url, stream=True)
    total_length = fetch_response_content_length(r) or 100
    chunk_size = 1024 * 1024

    with open(output_file_path.as_posix(), "wb") as f:
        with tqdm(total=total_length, unit="B", unit_scale=True, desc="Downloading sandbox project", ascii=True) as bar:
            for index, chunk in enumerate(r.iter_content(chunk_size=chunk_size)):
                f.write(chunk)
                bar.update(len(chunk))
                if progress_callback:
                    progress_callback(chunk_size * index / total_length)
            else:
                if progress_callback:
                    progress_callback(1.0)

    if not unpack:
        return output_file_path

    return unpack_archive(output_file_path, out_dir)


def unpack_archive(archive_path: Path, target_path: Path, delete=True):
    rich.print("Unpacking zip file. May take a bit.")
    shutil.unpack_archive(archive_path, target_path)
    if delete:
        os.remove(archive_path)
    return target_path
