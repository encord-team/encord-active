import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import requests
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlmodel import Session, select
from tqdm.auto import tqdm

from encord_active.db.models.project_data_unit_metadata import ProjectDataUnitMetadata
from encord_active.imports.local_files import download_to_local_file
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
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/f2140a72-c644-4c31-be66-3ef80b3718e5.json",
        hash=uuid.UUID("f2140a72-c644-4c31-be66-3ef80b3718e5"),
        name="[open-source][validation]-coco-2017-dataset",
        image_filename="coco.jpeg",
        stats=SandboxProjectStats(data_hash_count=4952, annotation_count=41420, class_count=80),
    ),
    uuid.UUID("d18819cb-2b75-4040-beb6-c63a901e6c84"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/d18819cb-2b75-4040-beb6-c63a901e6c84.json",
        hash=uuid.UUID("d18819cb-2b75-4040-beb6-c63a901e6c84"),
        name="[open-source]-covid-19-segmentations",
        image_filename="covid_segmentations.jpeg",
        stats=SandboxProjectStats(data_hash_count=100, annotation_count=588, class_count=13),
    ),
    uuid.UUID("5c96df58-beac-4e42-a74d-fce16622c5af"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/5c96df58-beac-4e42-a74d-fce16622c5af.json",
        hash=uuid.UUID("b37a48e0-6462-472d-baaa-2fcaf5ab9521"),
        name="[open-source][validation]-bdd-dataset",
        image_filename="bdd.jpeg",
        stats=SandboxProjectStats(data_hash_count=981, annotation_count=12983, class_count=8),
    ),
    uuid.UUID("5c96df58-beac-4e42-a74d-fce16622c5af"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/5c96df58-beac-4e42-a74d-fce16622c5af.json",
        hash=uuid.UUID("5c96df58-beac-4e42-a74d-fce16622c5af"),
        name="[open-source][test]-mnist-dataset",
        image_filename="mnist.png",
        stats=SandboxProjectStats(data_hash_count=70000, annotation_count=0, class_count=0),
    ),
    uuid.UUID("1f4752d7-4a7a-4c0e-8b08-dd4b1c5a8bc6"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/1f4752d7-4a7a-4c0e-8b08-dd4b1c5a8bc6.json",
        hash=uuid.UUID("1f4752d7-4a7a-4c0e-8b08-dd4b1c5a8bc6"),
        name="rareplanes",
        image_filename="rareplanes.jpeg",
        stats=SandboxProjectStats(data_hash_count=2710, annotation_count=6812, class_count=7),
    ),
    uuid.UUID("d6423838-f60e-41d9-b2ca-715aa2edef9c"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/d6423838-f60e-41d9-b2ca-715aa2edef9c.json",
        hash=uuid.UUID("d6423838-f60e-41d9-b2ca-715aa2edef9c"),
        name="quickstart",
        image_filename="quickstart.jpeg",
        stats=SandboxProjectStats(data_hash_count=199, annotation_count=1617, class_count=71),
    ),
}

ADDITIONAL_PROJECTS: dict[uuid.UUID, PrebuiltProject] = {
    uuid.UUID("aa2b21bd-6f2e-48fc-8f4f-4ba4d9b7bd67"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/aa2b21bd-6f2e-48fc-8f4f-4ba4d9b7bd67.json",
        hash=uuid.UUID("aa2b21bd-6f2e-48fc-8f4f-4ba4d9b7bd67"),
        name="[open-source][test]-limuc-ulcerative-colitis-classification",
        image_filename="limuc.png",
        stats=SandboxProjectStats(data_hash_count=1686, annotation_count=1686, class_count=4),
    ),
    # FIXME: some of the annotation have bad rle length
    # uuid.UUID("dc1cf137-f1b9-4c2f-973d-32512c971955"): PrebuiltProject(
    #     url="https://storage.googleapis.com/encord-active-sandbox-data/%5Bopen-source%5D-TACO-Official.zip",
    #     hash=uuid.UUID("dc1cf137-f1b9-4c2f-973d-32512c971955"),
    #     name="[open-source]-TACO-Official",
    #     image_filename="taco.jpeg",
    #     stats=SandboxProjectStats(data_hash_count=1500, annotation_count=5038, class_count=59),
    # ),
    uuid.UUID("14a52852-55f6-46c9-850a-40e11540605f"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/14a52852-55f6-46c9-850a-40e11540605f.json",
        hash=uuid.UUID("14a52852-55f6-46c9-850a-40e11540605f"),
        name="[open-source]-TACO-Unofficial",
        image_filename="taco_unofficial.jpeg",
        stats=SandboxProjectStats(data_hash_count=3731, annotation_count=8419, class_count=60),
    ),
    uuid.UUID("34413f3b-fed5-4a34-a279-b68a0c5fe325"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/34413f3b-fed5-4a34-a279-b68a0c5fe325.json",
        hash=uuid.UUID("34413f3b-fed5-4a34-a279-b68a0c5fe325"),
        name="[open-source][train]-mnist-dataset",
        image_filename="mnist.png",
        stats=SandboxProjectStats(data_hash_count=119000, annotation_count=0, class_count=0),
    ),
    uuid.UUID("d083bd28-fd4a-4b58-a80d-a1a9074d2cdc"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/d083bd28-fd4a-4b58-a80d-a1a9074d2cdc.json",
        hash=uuid.UUID("d083bd28-fd4a-4b58-a80d-a1a9074d2cdc"),
        name="[open-source][train]-Caltech-101",
        image_filename="caltech101_train.jpeg",
        stats=SandboxProjectStats(data_hash_count=5171, annotation_count=5171, class_count=101),
    ),
    uuid.UUID("c1679f72-6ad5-46f0-b011-f5b2c60e23d5"): PrebuiltProject(
        url="https://storage.googleapis.com/encord-active-sandbox-projects/project-dbs/c1679f72-6ad5-46f0-b011-f5b2c60e23d5.json",
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


def fetch_prebuilt_project(
    project: PrebuiltProject, engine: Engine, database_dir: Path, store_data_locally: bool = False
):
    r = requests.get(project.url, stream=True)
    total_length = fetch_response_content_length(r) or 100
    chunk_size = 1024 * 1024

    output_file_path = database_dir / f"{project.hash}.json"

    with open(output_file_path.as_posix(), "wb") as f:
        with tqdm(total=total_length, unit="B", unit_scale=True, desc="Downloading sandbox project", ascii=True) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))

    print("Building the database...")
    import_serialized_project(engine, output_file_path)
    output_file_path.unlink(missing_ok=False)

    if store_data_locally:
        with Session(engine) as sess:
            data_units = sess.exec(
                select(ProjectDataUnitMetadata).where(ProjectDataUnitMetadata.project_hash == project.hash)
            ).fetchall()
            project_dir = database_dir / str(project.hash)
            project_dir.mkdir(exist_ok=True)

            with ThreadPoolExecutor() as executor:
                for du in tqdm(data_units, desc="Downloading data units for local storage"):
                    if du.data_uri is None:
                        continue
                    file_path = project_dir / du.data_uri.split("/")[-1]
                    job = executor.submit(download_to_local_file, database_dir, file_path, du.data_uri)
                    du.data_uri = job.result()
                    sess.add(du)
            sess.commit()
