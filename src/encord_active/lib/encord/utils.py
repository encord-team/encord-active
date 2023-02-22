from json import dumps as json_dumps
from pathlib import Path
from typing import List, Optional

from encord import EncordUserClient
from encord.http.constants import RequestsSettings
from encord.orm.project import Project


def get_client(ssh_key_path: Path):
    return EncordUserClient.create_with_ssh_private_key(
        ssh_key_path.read_text(encoding="utf-8"), requests_settings=RequestsSettings(max_retries=5)
    )


def get_encord_projects(ssh_key_path: Path, query: Optional[str] = None) -> List[Project]:
    client = get_client(ssh_key_path)
    projects: List[Project] = list(map(lambda x: x["project"], client.get_projects(title_like=query)))
    return projects


def get_projects_json(ssh_key_path: Path, query: Optional[str] = None) -> str:
    projects = get_encord_projects(ssh_key_path, query)
    return json_dumps({p.project_hash: p.title for p in projects}, indent=2)
