from json import dumps as json_dumps
from typing import Optional

from encord import EncordUserClient


def get_projects_json(config, query: Optional[str] = None) -> str:
    ssh_key_path = config.get_or_query_ssh_key()
    client = EncordUserClient.create_with_ssh_private_key(ssh_key_path.read_text())
    projects = map(lambda x: x["project"], client.get_projects(title_like=query))
    return json_dumps({p.project_hash: p.title for p in projects}, indent=2)
