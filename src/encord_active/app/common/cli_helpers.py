from pathlib import Path

from encord import EncordUserClient, Project

from encord_active.lib.common.utils import fetch_project_meta


def get_local_project(project_dir: Path) -> Project:
    project_meta = fetch_project_meta(project_dir)

    ssh_key_path = Path(project_meta["ssh_key_path"])
    with open(ssh_key_path.expanduser(), "r", encoding="utf-8") as f:
        key = f.read()

    client = EncordUserClient.create_with_ssh_private_key(key)
    return client.get_project(project_meta.get("project_hash"))
