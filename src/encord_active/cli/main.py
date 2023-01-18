from pathlib import Path
from typing import List, Set, TypedDict

import rich
import typer
from rich.markup import escape
from rich.panel import Panel

import encord_active.app.conf  # pylint: disable=unused-import
from encord_active.cli.config import APP_NAME, config_cli
from encord_active.cli.imports import import_cli
from encord_active.cli.print import print_cli
from encord_active.cli.utils.decorators import bypass_streamlit_question, ensure_project
from encord_active.lib.project.local import NoFilesFoundError, ProjectExistsError

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


@cli.command(
    name="init",
)
def import_local_project(
    root: Path = typer.Argument(
        ...,
        help="The root directory of the dataset you are trying to import",
        file_okay=False,
    ),
    glob: List[str] = typer.Option(
        ["**/*.jpg", "**/*.png", "**/*.tiff"],
        "--glob",
        "-g",
        help='Glob pattern to choose files in the "leaf directories". Note that you can repeat the `--glob` argument if you want to match multiple things.',
    ),
    target: Path = typer.Option(
        Path.cwd(),
        "--target",
        "-t",
        help="Directory where the project would be saved.",
        file_okay=False,
    ),
    project_name: str = typer.Option(
        "",
        "--name",
        "-n",
        help="Name to give the new project. If no name is provided, the root directory will be used with '[EA] ' prepended",
    ),
    symlinks: bool = typer.Option(
        False,
        help="Use symlinks instead of copying images to the target directory.",
    ),
    dryrun: bool = typer.Option(
        False,
        help="Print the files that will be imported WITHOUT importing them.",
    ),
):
    """
    [bold]Initialise[/bold] a project from your local file system by searching for images based on the `glob` arguments.
    By default, all jpeg, jpg, png, and tiff files will be matched.


    """
    from encord_active.lib.project.local import init_local_project

    try:
        project_path = init_local_project(
            root=root, target=target, glob=glob, project_name=project_name, symlinks=symlinks, dryrun=dryrun
        )
        if dryrun:
            if not isinstance(project_path, list):
                raise ValueError("Expected a list of paths for the dryrun execution")
            file_paths = project_path

            directories: Set[Path] = set()
            for file in file_paths:
                directories.add(file.parent)
                print(file)
            print()
            rich.print(
                Panel(
                    f"Found {len(file_paths)} files in {len(directories)} directories.",
                    title=":bar_chart: Stats :bar_chart:",
                    expand=False,
                )
            )
        else:
            if not isinstance(project_path, Path):
                raise ValueError("Expected a single path for an actual init execution")

            cwd = Path.cwd()
            cd_dir = project_path.relative_to(cwd) if project_path.is_relative_to(cwd) else project_path

            panel = Panel(
                f"""
Project initialised :+1:

You can run

[cyan]cd "{escape(cd_dir.as_posix())}"
encord-active visualise[/cyan]

to open Encord Active and see your project.
            """,
                title="🌟 Success 🌟",
                style="green",
                expand=False,
            )
            rich.print(panel)

    except NoFilesFoundError as e:
        rich.print(
            Panel(
                e.args[0],
                title=":fire: No Files Found :fire:",
                expand=False,
                style="yellow",
            )
        )
        typer.Abort()
    except ProjectExistsError as e:
        rich.print(
            Panel(
                f"""
{e.args[0]}

Consider removing the directory or setting the `--name` option.
                """,
                title=":open_file_folder: Project already exists :open_file_folder:",
                expand=False,
                style="yellow",
            )
        )
        typer.Abort()


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
