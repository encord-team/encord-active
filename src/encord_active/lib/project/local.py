import json
from itertools import chain
from pathlib import Path
from typing import List, Optional, Union

import yaml
from encord.ontology import OntologyStructure
from tqdm.auto import tqdm

from encord_active.lib.encord.local_sdk import LocalUserClient
from encord_active.lib.metrics.execute import run_metrics

IMAGE_DATA_UNIT_FILENAME = "image_data_unit.json"


class NoFilesFoundError(Exception):
    """Exception raised when searching for files yielded an empty result."""

    def __init__(self):
        super().__init__("Couldn't find any files to import from the given specifications.")


class ProjectExistsError(Exception):
    """Exception raised when the target project already exists."""

    def __init__(self, project_path: Path):
        super().__init__(f"The project path `{project_path}` already exists.")
        self.project_path = project_path


def init_local_project(
    root: Path,
    target: Path,
    glob: Optional[List[str]] = None,
    project_name: str = "",
    symlinks: bool = False,
    dryrun: bool = False,
) -> Union[Path, List[Path]]:
    """
    Initialising an Encord Active project based on the data found from the `root`
    based on the `glob` arguments.

    Args:
        root: The root from which to search for files.
        target: The directory in which the new project will be stored.
        glob: A list of glob expressions that will be used to match image files.
        project_name: A specific project name to use. If left out, the project
            name will be the root directory name prepended with "[EA] "
        symlinks: If false, files will be copied. If true, files will be symlinked.
        dryrun: If false, a project will be created. If false, a list of all the
            matched files will be returned.

    Returns:
        The path to the project directory.
        If `dryrun` is True, a list of all the matched files will be returned
        without actually initialising a project.
    """
    if not project_name:
        project_name = f"[EA] {root.name}"

    project_path = target / project_name

    if project_path.is_dir():
        raise ProjectExistsError(project_path)

    if glob is None:
        glob = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.tiff"]

    files = list(chain(*[root.glob(g) for g in glob]))
    if dryrun:
        return files

    if not len(files):
        raise NoFilesFoundError()

    client = LocalUserClient(project_path)

    dataset = client.create_dataset(project_name, symlinks)
    for file in tqdm(files, desc="Importing data"):
        dataset.upload_image(file)

    empty_structure = OntologyStructure()
    ontology = client.create_ontology(title=project_name, description="", structure=empty_structure)
    project = client.create_project(
        project_title=project_name,
        description="",
        dataset_hashes=[dataset.dataset_hash],
        ontology_hash=ontology.ontology_hash,
    )
    project_dir = client.project_path
    ontology_file = project_dir / "ontology.json"
    ontology_file.write_text(json.dumps(project.ontology))

    project_meta = {
        "project_title": project.title,
        "project_description": project.description,
        "project_hash": project.project_hash,
    }
    meta_file_path = project_dir / "project_meta.yaml"
    meta_file_path.write_text(yaml.dump(project_meta), encoding="utf-8")

    label_row_meta_collection = {lr["label_hash"]: lr for lr in project.label_rows}
    label_row_meta_file_path = project_dir / "label_row_meta.json"
    label_row_meta_file_path.write_text(json.dumps(label_row_meta_collection, indent=2), encoding="utf-8")

    image_to_du = {}
    # Empty label rows for now
    for label_row_meta in tqdm(project.label_rows, desc="Constructing project"):
        label_row = project.create_label_row(label_row_meta["data_hash"])
        image_id = label_row["data_title"]  # This is specific to one image label rows
        for du in label_row["data_units"].values():
            data_hash = du["data_hash"]
            width = du["width"]
            height = du["height"]
            image_to_du[image_id] = {
                "data_hash": data_hash,
                "height": height,
                "width": width,
            }
        project.save_label_row(label_row["label_hash"], label_row)

    (project_dir / IMAGE_DATA_UNIT_FILENAME).write_text(json.dumps(image_to_du))

    run_metrics(data_dir=client.project_path, use_cache_only=True)
    return project_path
