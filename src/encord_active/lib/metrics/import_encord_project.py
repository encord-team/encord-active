import re
import sys
from pathlib import Path
from typing import Optional

import encord.exceptions
import rich
import typer
import yaml
from encord import EncordUserClient, Project
from rich.markup import escape
from rich.panel import Panel

from encord_active.app.app_config import AppConfig
from encord_active.app.app_print_utils import get_projects_json

project_root = Path(__file__).parents[1]
sys.path.append(project_root.as_posix())

from encord_active.lib.metrics.run_all import run_metrics

PROJECT_HASH_REGEX = r"([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})"


def main(config: AppConfig, encord_project_hash: Optional[str] = None):
    ssh_key_path = config.get_or_query_ssh_key()
    projects_root = config.get_or_query_project_path()

    client = EncordUserClient.create_with_ssh_private_key(ssh_key_path.read_text(encoding="utf-8"))

    if not encord_project_hash:
        # == Get project hash === #
        rich.print(
            Panel(
                """
    Encord Active will need to know the project hash associated with the project you
    wish to import. If you don't know this hash already, [bold green]you can enter "?"[/bold green] in the following
    prompt to get all your `project_hash`es and associated project titles listed.
        """,
                title="The Encord Project",
                expand=False,
            )
        )

    project: Optional[Project] = None
    if encord_project_hash:
        try:
            project = client.get_project(encord_project_hash)
            _ = project.title
        except encord.exceptions.AuthorisationError:
            rich.print("⚡️ [red]You don't have access to the project, sorry[/red] 😫")
            exit()
    else:
        while not encord_project_hash:
            _encord_project_hash = typer.prompt("Specify project hash").strip().lower()

            if _encord_project_hash == "?":
                rich.print(escape(get_projects_json(config)))

            if not re.match(PROJECT_HASH_REGEX, _encord_project_hash):
                rich.print("🙈 [orange]Project hash does not have the correct format.")
                rich.print("The format is a (hex) uuid: aaaaaaaa-bbbb-bbbb-bbbb-000000000000")
                rich.print("💡 Hint: You can type [green]?[/green] to list all your projects")
                continue

            project = client.get_project(_encord_project_hash)
            try:
                _ = project.title
            except encord.exceptions.AuthorisationError:
                rich.print("⚡️ [red]You don't have access to the project, sorry[/red] 😫")
                continue

            encord_project_hash = _encord_project_hash

    if project is None:
        exit()

    project_path = projects_root / project.title.replace(" ", "-")
    project_path.mkdir(exist_ok=True, parents=True)

    meta_data = {
        "project_title": project.title,
        "project_description": project.description,
        "project_hash": project.project_hash,
        "ssh_key_path": ssh_key_path.as_posix(),
    }
    meta_file_path = project_path / "project_meta.yaml"
    yaml_str = yaml.dump(meta_data)
    with meta_file_path.open("w", encoding="utf-8") as f:
        f.write(yaml_str)

    rich.print("Stored the following data:")
    rich.print(f"[magenta]{yaml_str}")
    rich.print(f"In file: [blue]`{escape(meta_file_path.as_posix())}`")
    rich.print()
    rich.print("Now downloading data and running metrics")

    run_metrics(data_dir=project_path)

    panel = Panel(
        """
The data is downloaded and the metrics are complete.

Now run

[cyan]encord-active visualise[/cyan]

to open Encord Active and see your project.
""",
        title="🌟 Success 🌟",
        style="green",
        expand=False,
    )
    rich.print(panel)
