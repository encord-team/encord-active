import os
import shutil
from pathlib import Path
from typing import Optional, TypedDict

import requests
import rich
import typer
from rich.markup import escape
from tqdm.auto import tqdm


class PrebuiltProject(TypedDict):
    url: str
    hash: str
    name: str


# GCP bucket links will be added here
PREBUILT_PROJECTS: dict[str, PrebuiltProject] = {
    "[open-source][validation]-coco-2017-dataset": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Bvalidation%5D-coco-2017-dataset.zip",
        hash="f2140a72-c644-4c31-be66-3ef80b3718e5",
        name="[open-source][validation]-coco-2017-dataset",
    ),
    "[open-source][test]-limuc-ulcerative-colitis-classification": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btest%5D-limuc-ulcerative-colitis-classification.zip",
        hash="aa2b21bd-6f2e-48fc-8f4f-4ba4d9b7bd67",
        name="[open-source][test]-limuc-ulcerative-colitis-classification",
    ),
    "[open-source]-covid-19-segmentations": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-covid-19-segmentations.zip",
        hash="d18819cb-2b75-4040-beb6-c63a901e6c84",
        name="[open-source]-covid-19-segmentations",
    ),
    "[open-source][validation]-bdd-dataset": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Bvalidation%5D-bdd-dataset.zip",
        hash="b37a48e0-6462-472d-baaa-2fcaf5ab9521",
        name="[open-source][validation]-bdd-dataset",
    ),
    "[open-source]-TACO-Official": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-TACO-Official.zip",
        hash="dc1cf137-f1b9-4c2f-973d-32512c971955",
        name="[open-source]-TACO-Official",
    ),
    "[open-source]-TACO-Unofficial": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-TACO-Unofficial.zip",
        hash="14a52852-55f6-46c9-850a-40e11540605f",
        name="[open-source]-TACO-Unofficial",
    ),
    "[open-source][train]-mnist-dataset": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btrain%5D-mnist-dataset.zip",
        hash="34413f3b-fed5-4a34-a279-b68a0c5fe325",
        name="[open-source][train]-mnist-dataset",
    ),
    "[open-source][test]-mnist-dataset": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btest%5D-mnist-dataset.zip",
        hash="5c96df58-beac-4e42-a74d-fce16622c5af",
        name="[open-source][test]-mnist-dataset",
    ),
    "rareplanes": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/rareplanes.zip",
        hash="1f4752d7-4a7a-4c0e-8b08-dd4b1c5a8bc6",
        name="rareplanes",
    ),
    "quickstart": PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/quickstart.zip",
        hash="d6423838-f60e-41d9-b2ca-715aa2edef9c",
        name="quickstart",
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


def fetch_prebuilt_project(project_name: str, out_dir: Path):
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
    total_length = fetch_response_content_length(r)
    with open(output_file_path.as_posix(), "wb") as f:
        with tqdm(total=total_length, unit="B", unit_scale=True, desc="Downloading sandbox project", ascii=True) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))

    rich.print("Unpacking zip file. May take a bit.")
    shutil.unpack_archive(output_file_path, out_dir)
    os.remove(output_file_path)
    return out_dir
