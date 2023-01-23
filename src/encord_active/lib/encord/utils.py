from json import dumps as json_dumps
from pathlib import Path
from typing import List, Optional

from encord import EncordUserClient
from encord.orm.project import Project


def get_encord_projects(ssh_key_path: Path, query: Optional[str] = None) -> List[Project]:
    client = EncordUserClient.create_with_ssh_private_key(ssh_key_path.read_text())
    projects: List[Project] = list(map(lambda x: x["project"], client.get_projects(title_like=query)))
    return projects


def get_projects_json(ssh_key_path: Path, query: Optional[str] = None) -> str:
    projects = get_encord_projects(ssh_key_path, query)
    return json_dumps({p.project_hash: p.title for p in projects}, indent=2)
