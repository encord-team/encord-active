from pathlib import Path
from typing import List, Optional, Set, TypedDict

import click
import rich
import typer
from rich.markup import escape
from rich.panel import Panel
from typer.core import TyperGroup

import encord_active.app.conf  # pylint: disable=unused-import
import encord_active.cli.utils.typer  # pylint: disable=unused-import
from encord_active.app.app_config import APP_NAME
from encord_active.cli.config import config_cli
from encord_active.cli.imports import import_cli
from encord_active.cli.print import print_cli
from encord_active.cli.utils.decorators import bypass_streamlit_question, ensure_project
from encord_active.cli.utils.prints import success_with_visualise_command
from encord_active.lib import constants as ea_constants
from encord_active.lib.metrics.execute import run_metrics, run_metrics_by_embedding_type
from encord_active.lib.metrics.heuristic.img_features import AreaMetric
from encord_active.lib.metrics.metric import EmbeddingType


class OrderedPanelGroup(TyperGroup):
    COMMAND_ORDER = [
        "quickstart",
        "download",
        "init",
        "import",
        "visualize",
        "metricize",
        "print",
        "config",
    ]

    def list_commands(self, ctx: click.Context):
        sorted_keys = [key for key in self.COMMAND_ORDER if key in self.commands.keys()]
        remaining = [key for key in self.commands.keys() if key not in self.COMMAND_ORDER]

        return sorted_keys + remaining


cli = typer.Typer(
    cls=OrderedPanelGroup,
    rich_markup_mode="rich",
    no_args_is_help=True,
    help=f"""
All commands in this CLI have a --help option, which will guide you on the way.
If you don't find the information you need here, we recommend that you visit
our main documentation: [blue]{ea_constants.DOCS_URL}[/blue]
""",
    epilog=f"""
Made by Encord. [bold]Get in touch[/bold]:


:call_me_hand: Slack Channel: [blue]{ea_constants.SLACK_URL}[/blue]

:e-mail: Email: [blue]{ea_constants.ENCORD_EMAIL}[/blue]

:star: Github: [blue]{ea_constants.GITHUB_URL}[/blue]
""",
)
cli.add_typer(config_cli, name="config", help="[green bold]Configure[/green bold] global settings 🔧")
cli.add_typer(import_cli, name="import", help="[green bold]Import[/green bold] Projects or Predictions ⬇️")
cli.add_typer(print_cli, name="print")


@cli.command()
def download(
    project_name: str = typer.Option(None, help="Name of the chosen project."),
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Directory where the project would be saved.", file_okay=False
    ),
):
    """
    [green bold]Download[/green bold] a sandbox dataset to get started. 📁

    * If --project_name is not given as an argument, available sandbox projects will be listed
     and you can select one from the menu.
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
            modified_project_name = project_name + (f" ({project_size} MB)" if project_size is not None else "")
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
    success_with_visualise_command(project_path, "Successfully downloaded sandbox dataset. ")


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
        ["**/*.jpg", "**/*.png", "**/*.jpeg", "**/*.tiff"],
        "--glob",
        "-g",
        help="Glob pattern to choose files. Note that you can repeat the `--glob` argument if you want to match multiple things.",
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
        help="Name to give the new project. If no name is provided, the root directory will be used with '[EA] ' prepended.",
    ),
    symlinks: bool = typer.Option(
        False,
        help="Use symlinks instead of copying images to the target directory.",
    ),
    dryrun: bool = typer.Option(
        False,
        help="Print the files that will be imported WITHOUT importing them.",
    ),
    metrics: bool = typer.Option(
        True,
        help="Run the metrics on the initiated project.",
    ),
):
    """
    [green bold]Initialize[/green bold] a project from your local file system :seedling:

    The command will search for images based on the `glob` arguments.

    By default, all jpeg, jpg, png, and tiff files will be matched.
    """
    from encord_active.lib.project.local import (
        NoFilesFoundError,
        ProjectExistsError,
        file_glob,
        init_local_project,
    )

    try:
        glob_result = file_glob(root, glob, images_only=True)
    except NoFilesFoundError as e:
        rich.print(
            Panel(
                str(e),
                title=":fire: No Files Found :fire:",
                expand=False,
                style="yellow",
            )
        )
        raise typer.Abort()

    if dryrun:
        directories: Set[Path] = set()
        rich.print("[blue]Matches:[/blue]")
        for file in glob_result.matched:
            directories.add(file.parent)
            rich.print(f"[blue]{escape(file.as_posix())}[/blue]")

        print()
        rich.print("[yellow]Excluded:[/yellow]")
        for file in glob_result.excluded:
            directories.add(file.parent)
            rich.print(f"[yellow]{escape(file.as_posix())}[/yellow]")

        print()
        rich.print(
            Panel(
                f"""
[blue]Found[/blue] {len(glob_result.matched)} file(s) in {len(directories)} directories.
[yellow]Excluded[/yellow] {len(glob_result.excluded)} file(s) because they do not seem to be images.
""",
                title=":bar_chart: Stats :bar_chart:",
                expand=False,
            )
        )
        raise typer.Abort()

    try:
        if not project_name:
            project_name = f"[EA] {root.name}"

        project_path = init_local_project(
            files=glob_result.matched, target=target, project_name=project_name, symlinks=symlinks
        )

        metricize_options = {"data_dir": project_path, "use_cache_only": True}
        if metrics:
            run_metrics_by_embedding_type(EmbeddingType.IMAGE, **metricize_options)
        else:
            # NOTE: we need to compute at  least one metric otherwise everything breaks
            run_metrics(filter_func=lambda x: isinstance(x, AreaMetric), **metricize_options)

        success_with_visualise_command(project_path, "Project initialised :+1:")

    except ProjectExistsError as e:
        rich.print(
            Panel(
                f"""
{str(e)}

Consider removing the directory or setting the `--name` option.
                """,
                title=":open_file_folder: Project already exists :open_file_folder:",
                expand=False,
                style="yellow",
            )
        )
        raise typer.Abort()


@cli.command(name="visualise", hidden=True)  # Alias for backward compatibility
@cli.command(name="visualize")
@bypass_streamlit_question
@ensure_project
def visualize(
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Path of the project you would like to visualise", file_okay=False
    ),
):
    """
    [green bold]Launch[/green bold] the application with the provided project ✨
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
    [green bold]Start[/green bold] Encord Active straight away 🏃💨
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
    metric_names: Optional[list[str]] = typer.Argument(None, help="Names of the metrics to run."),
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Directory of the project to run the metrics on.", file_okay=False
    ),
    run_all: bool = typer.Option(False, "--all", help="Run all available metrics."),
    fuzzy: bool = typer.Option(
        False, help="Enable fuzzy search in the selection. (press [TAB] to select more than one) 🪄"
    ),
):
    """
    [green bold]Execute[/green bold] metrics on your data and predictions 🧠
    """
    from InquirerPy import inquirer as i
    from InquirerPy.base.control import Choice

    from encord_active.lib.metrics.execute import (
        execute_metrics,
        get_metrics,
        load_metric,
    )

    metrics = list(map(load_metric, get_metrics()))
    if run_all:  # User chooses to run all available metrics
        selected_metrics = metrics

    # (interactive) User chooses some metrics via CLI prompt selection
    elif not metric_names:
        choices = list(map(lambda m: Choice(m, name=m.metadata.title), metrics))
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

    # (non-interactive) User chooses some metrics via --add (-a) option
    else:
        metric_name_to_cls = {m.metadata.title: m for m in metrics}
        used_metric_names = set()
        selected_metrics = []
        unknown_metric_names = []
        for name in metric_names:
            if name in used_metric_names:  # repeated metric name in --add
                continue
            used_metric_names.add(name)

            metric_cls = metric_name_to_cls.get(name, None)
            if metric_cls is None:  # unknown and/or wrong metric name in user selection
                unknown_metric_names.append(name)
            else:
                selected_metrics.append(metric_cls)

        if len(unknown_metric_names) > 0:
            rich.print("No available metric has this name:")
            for name in unknown_metric_names:
                rich.print(f"[yellow]{name}[/yellow]")
            raise typer.Abort()

    execute_metrics(selected_metrics, data_dir=target, use_cache_only=True)


@cli.command(rich_help_panel="Resources")
def docs():
    """
    [green bold]Read[/green bold] the documentation :book:
    """
    import webbrowser

    webbrowser.open(ea_constants.DOCS_URL)


@cli.command(name="join-slack", rich_help_panel="Resources")
def join_slack():
    """
    [green bold]Join[/green bold] the Slack community :family:
    """
    import webbrowser

    webbrowser.open(ea_constants.SLACK_INVITE_URL)


@cli.callback(invoke_without_command=True)
def version(version_: bool = typer.Option(False, "--version", "-v", help="Print the current version of Encord Active")):
    if version_:
        from encord_active import __version__ as ea_version

        rich.print(f"Version: [green]{ea_version}[/green]")
        exit()


if __name__ == "__main__":
    cli(prog_name=APP_NAME)
