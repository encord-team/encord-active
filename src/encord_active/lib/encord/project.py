from json import dumps as json_dumps
from pathlib import Path
from typing import Optional

from encord import EncordUserClient


def get_projects_json(ssh_key_path: Path, query: Optional[str] = None) -> str:
    client = EncordUserClient.create_with_ssh_private_key(ssh_key_path.read_text())
    projects = map(lambda x: x["project"], client.get_projects(title_like=query))
    return json_dumps({p.project_hash: p.title for p in projects}, indent=2)
