import sys
from pathlib import Path
from typing import Optional

import inquirer as i
import rich
import typer

import encord_active.app.conf  # pylint: disable=unused-import
from encord_active.app.cli.config import APP_NAME, app_config, config_cli
from encord_active.app.cli.imports import import_cli
from encord_active.app.cli.print import print_cli

cli = typer.Typer(
    rich_markup_mode="rich",
    help="""
All commands in this CLI have a --help option, which will guide you on the way.
If you don't find the information you need here, we recommend that you visit
our main documentation: [blue]https://encord-active-docs.web.app[/blue]
""",
    epilog="""
Made by Encord. Contact Encord here: [blue]https://encord.com/contact_us/[/blue] to learn more
about our active learning platform for computer vision.
""",
)
cli.add_typer(config_cli, name="config", help="Configure global settings 🔧")
cli.add_typer(import_cli, name="import", help="Import Projects or Predictions ⬇️")
cli.add_typer(print_cli, name="print")


@cli.command()
def download():
    """
    Try out Encord Active fast. [bold]Download[/bold] an existing dataset to get started. 📁

    * Available prebuilt projects will be listed and the user can select one from the menu.
    """
    project_parent_dir = app_config.get_or_query_project_path()

    from encord_active.lib.metrics.fetch_prebuilt_metrics import (
        PREBUILT_PROJECTS,
        fetch_prebuilt_project_size,
    )

    rich.print("Loading prebuilt projects ...")
    project_names_with_storage = []
    for project_name in PREBUILT_PROJECTS.keys():
        project_size = fetch_prebuilt_project_size(project_name)
        modified_project_name = project_name + (f" ({project_size} mb)" if project_size is not None else "")
        project_names_with_storage.append(modified_project_name)

    questions = [i.List("project_name", message="Choose a project", choices=project_names_with_storage)]
    answers = i.prompt(questions)
    if not answers or "project_name" not in answers:
        rich.print("No project was selected.")
        raise typer.Abort()
    project_name = answers["project_name"].split(" ", maxsplit=1)[0]

    # create project folder
    project_dir = project_parent_dir / project_name
    project_dir.mkdir(exist_ok=True)

    from encord_active.lib.metrics.fetch_prebuilt_metrics import fetch_prebuilt_project

    fetch_prebuilt_project(project_name, project_dir)


@cli.command()
def visualise(
    project_path: Optional[Path] = typer.Argument(
        None,
        help="Path of the project you would like to visualise",
        file_okay=False,
    ),
):
    """
    Launches the application with the provided project ✨
    """
    from encord_active.app.common.cli_helpers import choose_local_project

    project_path = project_path or choose_local_project(app_config)

    if not project_path:
        raise typer.Abort()

    streamlit_page = (Path(__file__).parents[2] / "streamlit_entrypoint.py").expanduser().absolute()
    data_dir = project_path.expanduser().absolute().as_posix()
    sys.argv = ["streamlit", "run", streamlit_page.as_posix(), data_dir]

    from streamlit.web import cli as stcli

    sys.exit(stcli.main())  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli(prog_name=APP_NAME)
