from pathlib import Path
from typing import Optional

import encord.exceptions
import rich
import yaml
from rich.markup import escape
from rich.panel import Panel

from encord_active.lib.encord.utils import get_client, get_encord_projects
from encord_active.lib.metrics.execute import run_metrics
from encord_active.lib.metrics.io import fill_metrics_meta_with_builtin_metrics
from encord_active.lib.metrics.metadata import update_metrics_meta
from encord_active.lib.project.project_file_structure import ProjectFileStructure

PROJECT_HASH_REGEX = r"([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})"


def import_encord_project(ssh_key_path: Path, target: Path, encord_project_hash: Optional[str]) -> Path:
    client = get_client(ssh_key_path)

    if encord_project_hash:
        project_hash = encord_project_hash
    else:
        from InquirerPy import inquirer as i
        from InquirerPy.base.control import Choice

        projects = get_encord_projects(ssh_key_path)
        if not projects:
            rich.print(
                Panel(
                    """
Couldn't find any projects to import.
Check that you have the correct ssh key set up and available projects on [blue]https://app.encord.com/projects[/blue].
                    """,
                    title="⚡️ [red]Failed to find projects[/red] ⚡️",
                    style="yellow",
                    expand=False,
                )
            )
            exit()

        choices = list(map(lambda p: Choice(p.project_hash, name=p.title), projects))
        project_hash = i.fuzzy(
            message="What project would you like to import?",
            choices=choices,
            vi_mode=True,
            multiselect=False,
            instruction="💡 Type a (fuzzy) search query to find the project you want to import.",
        ).execute()

    try:
        project = client.get_project(project_hash)
        _ = project.title
    except encord.exceptions.AuthorisationError:
        rich.print("⚡️ [red]You don't have access to the project, sorry[/red] 😫")
        exit()

    project_path = target / project.title.replace(" ", "-")
    project_path.mkdir(exist_ok=True, parents=True)
    project_file_structure = ProjectFileStructure(project_path)

    meta_data = {
        "project_title": project.title,
        "project_description": project.description,
        "project_hash": project.project_hash,
        "ssh_key_path": ssh_key_path.as_posix(),
        "has_remote": True,
    }
    yaml_str = yaml.dump(meta_data)
    project_file_structure.project_meta.write_text(yaml_str, encoding="utf-8")

    # attach builtin metrics to the project
    metrics_meta = fill_metrics_meta_with_builtin_metrics()
    update_metrics_meta(project_file_structure, metrics_meta)

    rich.print("Stored the following data:")
    rich.print(f"[magenta]{escape(yaml_str)}")
    rich.print(f'In file: [blue]"{escape(project_file_structure.project_meta.as_posix())}"')
    rich.print()
    rich.print("Now downloading data and running metrics")

    run_metrics(data_dir=project_path)

    return project_path
