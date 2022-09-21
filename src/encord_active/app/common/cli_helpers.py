from pathlib import Path
from typing import Optional

import inquirer as i
import rich
from encord import EncordUserClient, Project

from encord_active.app.app_config import AppConfig, ConfigProperties
from encord_active.lib.common.utils import fetch_project_meta


def choose_local_project(config: AppConfig) -> Optional[Path]:
    projects_dir = config.contents.get(ConfigProperties.PROJECTS_DIR.value)
    if not projects_dir:
        rich.print("[bold red] Projects dir is not defined[/bold] (Probably means you haven't imported a project).")
        return None
    else:
        projects_dir = Path(projects_dir)

    project_names = [p.resolve() for p in projects_dir.glob("*") if p.is_dir() and (p / "project_meta.yaml").is_file()]
    questions = [i.List("project_name", message="Choose a project", choices=project_names)]
    answers = i.prompt(questions)
    if not answers or "project_name" not in answers:
        return None

    selected_project = answers["project_name"]
    return projects_dir / selected_project


def get_local_project(project_dir: Path) -> Project:
    project_meta = fetch_project_meta(project_dir)

    ssh_key_path = Path(project_meta["ssh_key_path"])
    with open(ssh_key_path.expanduser(), "r", encoding="utf-8") as f:
        key = f.read()

    client = EncordUserClient.create_with_ssh_private_key(key)
    return client.get_project(project_meta.get("project_hash"))
