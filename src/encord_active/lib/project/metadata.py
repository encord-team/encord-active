from pathlib import Path
from typing import Optional, TypedDict

import yaml
from encord import Project as EncordProject

from encord_active.lib.encord.utils import get_client


class ProjectMeta(TypedDict):
    project_description: str
    project_hash: str
    project_title: str
    ssh_key_path: str
    has_remote: Optional[bool]
    data_version: int
    store_data_locally: bool


class ProjectNotFound(Exception):
    """Exception raised when a path doesn't contain a valid project.

    Attributes:
        project_dir -- path to a project directory
    """

    def __init__(self, project_dir):
        self.project_dir = project_dir
        super().__init__(f"Couldn't find meta file for project in `{project_dir}`")


def fetch_project_meta(data_dir: Path) -> ProjectMeta:
    meta_file = data_dir / "project_meta.yaml"
    if not meta_file.is_file():
        raise ProjectNotFound(data_dir)

    with meta_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def update_project_meta(data_dir: Path, updated_meta: ProjectMeta):
    meta_file_path = data_dir / "project_meta.yaml"
    meta_file_path.write_text(yaml.safe_dump(updated_meta), encoding="utf-8")


def fetch_encord_project_instance(data_dir: Path) -> EncordProject:
    project_meta = fetch_project_meta(data_dir)
    # == Key file == #
    if "ssh_key_path" not in project_meta:
        raise ValueError("SSH Key path missing in project metadata.")

    private_key_file = Path(project_meta["ssh_key_path"]).expanduser().absolute()
    client = get_client(private_key_file)

    # == Project hash == #
    if "project_hash" not in project_meta:
        raise ValueError("`project_hash` is missing in project metadata.")
    project_hash = project_meta["project_hash"]
    project = client.get_project(project_hash=project_hash)
    return project
