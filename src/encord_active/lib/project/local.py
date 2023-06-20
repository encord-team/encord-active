import json
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import List, Optional

import yaml
from encord.ontology import OntologyStructure
from tqdm.auto import tqdm

from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.db.prisma_init import ensure_prisma_db
from encord_active.lib.encord.local_sdk import (
    FileTypeNotSupportedError,
    LocalUserClient,
    get_mimetype,
)
from encord_active.lib.encord.utils import handle_enum_and_datetime
from encord_active.lib.labels.label_transformer import (
    LabelTransformer,
    LabelTransformerWrapper,
)
from encord_active.lib.metrics.io import fill_metrics_meta_with_builtin_metrics
from encord_active.lib.metrics.metadata import update_metrics_meta
from encord_active.lib.project.project_file_structure import ProjectFileStructure


class NoFilesFoundError(Exception):
    """Exception raised when searching for files yielded an empty result."""

    def __init__(self):
        super().__init__("Couldn't find any files to import from the given specifications.")


class ProjectExistsError(Exception):
    """Exception raised when the target project already exists."""

    def __init__(self, project_path: Path):
        super().__init__(f"The project path `{project_path}` already exists.")
        self.project_path = project_path


@dataclass
class GlobResult:
    matched: List[Path]
    excluded: List[Path] = field(default_factory=list)


def file_glob(root: Path, glob: List[str], images_only: bool = False) -> GlobResult:
    files = list(chain(*[root.glob(g) for g in glob]))

    if not len(files):
        raise NoFilesFoundError()

    if images_only:
        matches = []
        excluded = []
        for file in files:
            if "image" in get_mimetype(file):
                matches.append(file)
            else:
                excluded.append(file)
        return GlobResult(matches, excluded)

    return GlobResult(files)


def init_local_project(
    files: List[Path],
    target: Path,
    project_name: str = "",
    symlinks: bool = False,
    label_transformer: Optional[LabelTransformer] = None,
    label_paths: Optional[List[Path]] = None,
) -> Path:
    """
    Initialising an Encord Active project based on the data found from the `root`
    based on the `glob` arguments.

    Args:
        files: The file paths to include in the project.
        target: The directory in which the new project will be initialised.
        project_name: A specific project name to use. If left out, the project
            name will be the root directory name prepended with "[EA] "
        symlinks: If false, files will be copied. If true, files will be symlinked.

    Returns:
        The path to the project directory.
        If `dryrun` is True, a list of all the matched files will be returned
        without actually initialising a project.
    """
    project_path = target / project_name

    if project_path.is_dir():
        raise ProjectExistsError(project_path)

    client = LocalUserClient(project_path)

    dataset = client.create_dataset(project_name, symlinks)
    for file in tqdm(files, desc="Importing data"):
        try:
            dataset.upload_image(file)
        except FileTypeNotSupportedError:
            print(f"{file} will be skipped as it doesn't seem to be an image.")

    empty_structure = OntologyStructure()
    ontology = client.create_ontology(title=project_name, description="", structure=empty_structure)
    project = client.create_project(
        project_title=project_name,
        description="",
        dataset_hashes=[dataset.dataset_hash],
        ontology_hash=ontology.ontology_hash,
    )
    project_file_structure = ProjectFileStructure(project_path)

    project_meta = {
        "project_title": project.title,
        "project_description": project.description,
        "project_hash": project.project_hash,
        "has_remote": False,
    }
    project_file_structure.project_meta.write_text(yaml.dump(project_meta), encoding="utf-8")

    # attach builtin metrics to the project
    metrics_meta = fill_metrics_meta_with_builtin_metrics()
    update_metrics_meta(project_file_structure, metrics_meta)

    label_row_meta_collection = {lr.label_hash: handle_enum_and_datetime(lr) for lr in project.label_row_meta}
    project_file_structure.label_row_meta.write_text(json.dumps(label_row_meta_collection, indent=2), encoding="utf-8")

    image_to_du = {}
    for label_row_meta in tqdm(project.label_row_meta, desc="Constructing project"):
        label_row = project.create_label_row(label_row_meta.data_hash)
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

    transformer = LabelTransformerWrapper(ontology.structure, project.label_rows, label_transformer)
    transformer.add_labels(label_paths or [], data_paths=files)

    ensure_prisma_db(project_file_structure.prisma_db)

    with PrismaConnection(project_file_structure) as conn:
        batcher = conn.batch_()
        for label_row in tqdm(project.label_rows, desc="Saving label rows"):
            project.save_label_row(label_row["label_hash"], label_row, batcher)
        batcher.commit()

    project_file_structure.image_data_unit.write_text(json.dumps(image_to_du))
    project_file_structure.ontology.write_text(json.dumps(project.ontology))

    return project_path
