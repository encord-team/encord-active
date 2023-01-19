from pathlib import Path
from typing import List, TypedDict

import rich
import typer
from rich.markup import escape
from rich.panel import Panel

import encord_active.app.conf  # pylint: disable=unused-import
from encord_active.cli.config import APP_NAME, config_cli
from encord_active.cli.imports import import_cli
from encord_active.cli.print import print_cli
from encord_active.cli.utils.decorators import bypass_streamlit_question, ensure_project

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
def download(
    project_name: str = typer.Option(None, help="Name of the chosen project."),
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Directory where the project would be saved.", file_okay=False
    ),
):
    """
    Try out Encord Active fast. [bold]Download[/bold] an existing dataset to get started. 📁

    * If --project_name is not given as an argument, available prebuilt projects will be listed
     and the user can select one from the menu.
    """
    from InquirerPy import inquirer as i

    from encord_active.lib.project.sandbox_projects import (
        PREBUILT_PROJECTS,
        fetch_prebuilt_project_size,
    )

    if project_name is not None and project_name not in PREBUILT_PROJECTS:
        rich.print("No such project in prebuilt projects.")
        raise typer.Abort()

    if not project_name:
        rich.print("Loading prebuilt projects ...")
        project_names_with_storage = []
        for project_name in PREBUILT_PROJECTS.keys():
            project_size = fetch_prebuilt_project_size(project_name)
            modified_project_name = project_name + (f" ({project_size} mb)" if project_size is not None else "")
            project_names_with_storage.append(modified_project_name)

        answer = i.select(message="Choose a project", choices=project_names_with_storage, vi_mode=True).execute()
        if not answer:
            rich.print("No project was selected.")
            raise typer.Abort()
        project_name = answer.split(" ", maxsplit=1)[0]

    # create project folder
    project_dir = target / project_name
    project_dir.mkdir(exist_ok=True)

    from encord_active.lib.project.sandbox_projects import fetch_prebuilt_project

    project_path = fetch_prebuilt_project(project_name, project_dir)

    cwd = Path.cwd()
    cd_dir = project_path.relative_to(cwd) if project_path.is_relative_to(cwd) else project_path

    rich.print(
        Panel(
            f"""
Successfully downloaded sandbox dataset. To view the data, run:

[cyan]cd "{escape(cd_dir.as_posix())}"
encord-active visualise
    """,
            title="🌟 Success 🌟",
            style="green",
            expand=False,
        )
    )


@cli.command()
@bypass_streamlit_question
@ensure_project
def visualise(
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Path of the project you would like to visualise", file_okay=False
    ),
):
    """
    Launches the application with the provided project ✨
    """
    from encord_active.cli.utils.streamlit import launch_streamlit_app

    launch_streamlit_app(target)


@cli.command()
@bypass_streamlit_question
def quickstart(
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Directory where the project would be saved.", file_okay=False
    ),
):
    """
    Take the shortcut and start the application straight away 🏃💨
    """
    from encord_active.cli.utils.streamlit import launch_streamlit_app
    from encord_active.lib.project.sandbox_projects import fetch_prebuilt_project

    project_name = "quickstart"
    project_dir = target / project_name
    project_dir.mkdir(exist_ok=True)

    fetch_prebuilt_project(project_name, project_dir)
    launch_streamlit_app(project_dir)


@cli.command()
@ensure_project
def metricize(
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Directory of the project to run the metrics on.", file_okay=False
    ),
    fuzzy: bool = typer.Option(
        False, help="Enbale fuzzy search in the selection. (press [TAB] to select more than one) 🪄"
    ),
):
    """
    Execute metrics on your data and predictions 🧠
    """
    from InquirerPy import inquirer as i
    from InquirerPy.base.control import Choice

    from encord_active.lib.metrics.execute import (
        execute_metrics,
        get_metrics,
        load_metric,
    )

    metrics = list(map(load_metric, get_metrics()))
    choices = list(map(lambda m: Choice(m, name=m.TITLE), metrics))
    Options = TypedDict("Options", {"message": str, "choices": List[Choice], "vi_mode": bool})
    options: Options = {
        "message": "What metrics would you like to run?",
        "choices": choices,
        "vi_mode": True,
    }

    if fuzzy:
        options["message"] += " [blue](press [TAB] to select more than one)[/blue]"
        selected_metrics = i.fuzzy(**options, multiselect=True).execute()
    else:
        selected_metrics = i.checkbox(**options).execute()

    execute_metrics(selected_metrics, data_dir=target, use_cache_only=True)


if __name__ == "__main__":
    cli(prog_name=APP_NAME)
