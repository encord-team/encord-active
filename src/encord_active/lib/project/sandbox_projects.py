import os
import shutil
from pathlib import Path
from typing import Optional

import requests
import rich
import typer
from rich.markup import escape
from tqdm.auto import tqdm

# GCP bucket links will be added here
PREBUILT_PROJECTS = {
    "[open-source][validation]-coco-2017-dataset": "https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Bvalidation%5D-coco-2017-dataset.zip",
    "[open-source][test]-limuc-ulcerative-colitis-classification": "https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btest%5D-limuc-ulcerative-colitis-classification.zip",
    "[open-source]-covid-19-segmentations": "https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-covid-19-segmentations.zip",
    "[open-source][validation]-bdd-dataset": "https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Bvalidation%5D-bdd-dataset.zip",
    "[open-source]-TACO-Official": "https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-TACO-Official.zip",
    "[open-source]-TACO-Unofficial": "https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-TACO-Unofficial.zip",
    "rareplanes": "https://storage.googleapis.com/encord-active-sandbox-data/rareplanes.zip",
    "quickstart": "https://storage.googleapis.com/encord-active-sandbox-data/quickstart.zip",
}


def fetch_prebuilt_project_size(project_name: str) -> Optional[float]:
    """
    Returns the storage size (MB) of a prebuilt project stored in the cloud

    :param project_name: Prebuilt project name
    :return: Prebuilt project's storage size (MB). Can return None if the information is not available.
    """
    if project_name not in PREBUILT_PROJECTS.keys():
        return None
    url = PREBUILT_PROJECTS[project_name]
    with requests.head(url) as r:
        total_length = fetch_response_content_length(r)
        size = round(total_length / 1000000.0, 1) if total_length is not None else None
    return size


def fetch_response_content_length(r: requests.Response) -> Optional[int]:
    return int(r.headers["content-length"]) if "content-length" in r.headers.keys() else None


def fetch_prebuilt_project(project_name: str, out_dir: Path):
    url = PREBUILT_PROJECTS[project_name]
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
