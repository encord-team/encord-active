import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional

import requests
import typer
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from encord_active.imports.sandbox.deserialize import import_serialized_project
from encord_active.server.settings import AvailableSandboxProjects


class SandboxProjectStats(BaseModel):
    data_hash_count: int
    annotation_count: int
    class_count: int


class PrebuiltProject(BaseModel):
    url: str
    hash: uuid.UUID
    name: str
    image_filename: str
    stats: SandboxProjectStats


IMAGES_PATH = (Path(__file__).parent).resolve() / "images"

BASE_PROJECTS: dict[uuid.UUID, PrebuiltProject] = {
    uuid.UUID("f2140a72-c644-4c31-be66-3ef80b3718e5"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Bvalidation%5D-coco-2017-dataset.zip",
        hash=uuid.UUID("f2140a72-c644-4c31-be66-3ef80b3718e5"),
        name="[open-source][validation]-coco-2017-dataset",
        image_filename="coco.jpeg",
        stats=SandboxProjectStats(data_hash_count=4952, annotation_count=41420, class_count=80),
    ),
    uuid.UUID("d18819cb-2b75-4040-beb6-c63a901e6c84"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-covid-19-segmentations.zip",
        hash=uuid.UUID("d18819cb-2b75-4040-beb6-c63a901e6c84"),
        name="[open-source]-covid-19-segmentations",
        image_filename="covid_segmentations.jpeg",
        stats=SandboxProjectStats(data_hash_count=100, annotation_count=588, class_count=13),
    ),
    uuid.UUID("5c96df58-beac-4e42-a74d-fce16622c5af"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Bvalidation%5D-bdd-dataset.zip",
        hash=uuid.UUID("b37a48e0-6462-472d-baaa-2fcaf5ab9521"),
        name="[open-source][validation]-bdd-dataset",
        image_filename="bdd.jpeg",
        stats=SandboxProjectStats(data_hash_count=981, annotation_count=12983, class_count=8),
    ),
    uuid.UUID("5c96df58-beac-4e42-a74d-fce16622c5af"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btest%5D-mnist-dataset.zip",
        hash=uuid.UUID("5c96df58-beac-4e42-a74d-fce16622c5af"),
        name="[open-source][test]-mnist-dataset",
        image_filename="mnist.png",
        stats=SandboxProjectStats(data_hash_count=70000, annotation_count=0, class_count=0),
    ),
    uuid.UUID("1f4752d7-4a7a-4c0e-8b08-dd4b1c5a8bc6"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/rareplanes.zip",
        hash=uuid.UUID("1f4752d7-4a7a-4c0e-8b08-dd4b1c5a8bc6"),
        name="rareplanes",
        image_filename="rareplanes.jpeg",
        stats=SandboxProjectStats(data_hash_count=2710, annotation_count=6812, class_count=7),
    ),
    uuid.UUID("d6423838-f60e-41d9-b2ca-715aa2edef9c"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/quickstart.zip",
        hash=uuid.UUID("d6423838-f60e-41d9-b2ca-715aa2edef9c"),
        name="quickstart",
        image_filename="quickstart.jpeg",
        stats=SandboxProjectStats(data_hash_count=199, annotation_count=1617, class_count=71),
    ),
}

ADDITIONAL_PROJECTS: dict[uuid.UUID, PrebuiltProject] = {
    uuid.UUID("aa2b21bd-6f2e-48fc-8f4f-4ba4d9b7bd67"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btest%5D-limuc-ulcerative-colitis-classification.zip",
        hash=uuid.UUID("aa2b21bd-6f2e-48fc-8f4f-4ba4d9b7bd67"),
        name="[open-source][test]-limuc-ulcerative-colitis-classification",
        image_filename="limuc.png",
        stats=SandboxProjectStats(data_hash_count=1686, annotation_count=1686, class_count=4),
    ),
    uuid.UUID("dc1cf137-f1b9-4c2f-973d-32512c971955"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-TACO-Official.zip",
        hash=uuid.UUID("dc1cf137-f1b9-4c2f-973d-32512c971955"),
        name="[open-source]-TACO-Official",
        image_filename="taco.jpeg",
        stats=SandboxProjectStats(data_hash_count=1500, annotation_count=5038, class_count=59),
    ),
    uuid.UUID("14a52852-55f6-46c9-850a-40e11540605f"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-TACO-Unofficial.zip",
        hash=uuid.UUID("14a52852-55f6-46c9-850a-40e11540605f"),
        name="[open-source]-TACO-Unofficial",
        image_filename="taco_unofficial.jpeg",
        stats=SandboxProjectStats(data_hash_count=3731, annotation_count=8419, class_count=60),
    ),
    uuid.UUID("34413f3b-fed5-4a34-a279-b68a0c5fe325"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btrain%5D-mnist-dataset.zip",
        hash=uuid.UUID("34413f3b-fed5-4a34-a279-b68a0c5fe325"),
        name="[open-source][train]-mnist-dataset",
        image_filename="mnist.png",
        stats=SandboxProjectStats(data_hash_count=119000, annotation_count=0, class_count=0),
    ),
    uuid.UUID("d083bd28-fd4a-4b58-a80d-a1a9074d2cdc"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btrain%5D-Caltech-101.zip",
        hash=uuid.UUID("d083bd28-fd4a-4b58-a80d-a1a9074d2cdc"),
        name="[open-source][train]-Caltech-101",
        image_filename="caltech101_train.jpeg",
        stats=SandboxProjectStats(data_hash_count=5171, annotation_count=5171, class_count=101),
    ),
    uuid.UUID("c1679f72-6ad5-46f0-b011-f5b2c60e23d5"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D%5Btest%5D-Caltech-101.zip",
        hash=uuid.UUID("c1679f72-6ad5-46f0-b011-f5b2c60e23d5"),
        name="[open-source][test]-Caltech-101",
        image_filename="caltech101_test.jpeg",
        stats=SandboxProjectStats(data_hash_count=3506, annotation_count=3506, class_count=101),
    ),
}

ALL_SANDBOX_PROEJCTS = {**BASE_PROJECTS, **ADDITIONAL_PROJECTS}


def available_sandbox_projects(
    sandbox_projects: AvailableSandboxProjects = AvailableSandboxProjects.ALL,
) -> Dict[uuid.UUID, PrebuiltProject]:
    if sandbox_projects == AvailableSandboxProjects.ALL:
        return {**BASE_PROJECTS, **ADDITIONAL_PROJECTS}
    elif sandbox_projects == AvailableSandboxProjects.BASE:
        return BASE_PROJECTS
    else:
        return {}


def fetch_prebuilt_project_size(project_hash: uuid.UUID) -> Optional[float]:
    """
    Returns the storage size (MB) of a prebuilt project stored in the cloud

    :param project_hash: Prebuilt project identifier
    :return: Prebuilt project's storage size (MB). Can return None if the information is not available.
    """
    if project_hash not in ALL_SANDBOX_PROEJCTS:
        return None
    url = ALL_SANDBOX_PROEJCTS[project_hash].url
    with requests.head(url) as r:
        total_length = fetch_response_content_length(r)
        size = round(total_length / 1000000.0, 1) if total_length is not None else None
    return size


def fetch_response_content_length(r: requests.Response) -> Optional[int]:
    return int(r.headers["content-length"]) if "content-length" in r.headers.keys() else None


def fetch_prebuild_project(project_hash: uuid.UUID, engine: Engine, database_dir: Path) -> None:
    local_data = database_dir / "local-data"
    print("WARNING: using DEV mode for prebuild project as remote files have not been updated")
    dev_path_serialized_folder = (
        Path(__file__).parent.parent.parent.parent.parent / "local_tests" / "migrate_all_quickstart"
    )
    print(f"DEBUG: using {dev_path_serialized_folder} as dev serialized project path")
    quickstart_folder = dev_path_serialized_folder / f"export-{project_hash}"
    store_folder = local_data / f"export-{project_hash}"
    if not quickstart_folder.exists() or not quickstart_folder.is_dir():
        raise RuntimeError(
            f"Missing valid dev-mode prebuilt for project: {quickstart_folder}"
            f"(exists={quickstart_folder.exists()}, is_dir={quickstart_folder.is_dir()})"
        )
    db_json_file = store_folder / "db.json"
    if store_folder.exists():
        re_download = typer.confirm("Do you want to re-download the project?")
        if not re_download:
            return
        if db_json_file.exists():
            db_json_file.unlink(missing_ok=False)
        shutil.copytree(quickstart_folder, store_folder, dirs_exist_ok=True)
    else:
        shutil.copytree(quickstart_folder, store_folder)
    import_serialized_project(engine, db_json_file)
    db_json_file.unlink(missing_ok=False)


"""
def fetch_prebuilt_project(
    project_name: str, out_dir: Path, *, unpack=True, progress_callback: Optional[Callable] = None
):
    url = available_prebuilt_projects()[project_name]["url"]
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

    unpacked_archive_path = unpack_archive(output_file_path, out_dir)
    ensure_safe_project(unpacked_archive_path)
    return unpacked_archive_path


def unpack_archive(archive_path: Path, target_path: Path, delete=True):
    rich.print("Unpacking zip file. May take a bit.")
    shutil.unpack_archive(archive_path, target_path)
    if delete:
        os.remove(archive_path)
    return target_path
"""
