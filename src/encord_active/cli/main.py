import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import click
import rich
import typer
from dotenv import load_dotenv
from rich.markup import escape
from rich.panel import Panel
from typer.core import TyperGroup

from encord_active.cli.project import project_cli
from encord_active.cli.utils.server import ensure_safe_project

load_dotenv()

import encord_active.cli.utils.typer  # pylint: disable=unused-import
import encord_active.db.models as __fixme_debugging
import encord_active.lib.db  # pylint: disable=unused-import
from encord_active.cli.app_config import APP_NAME
from encord_active.cli.config import config_cli
from encord_active.cli.imports import import_cli
from encord_active.cli.metric import metric_cli
from encord_active.cli.print import print_cli
from encord_active.cli.utils.decorators import ensure_project, find_child_projects
from encord_active.cli.utils.prints import success_with_vizualise_command
from encord_active.lib import constants as ea_constants
from encord_active.lib.common.module_loading import ModuleLoadError
from encord_active.lib.project.metadata import fetch_project_meta


class OrderedPanelGroup(TyperGroup):
    COMMAND_ORDER = [
        "quickstart",
        "download",
        "start",
        "init",
        "import",
        "refresh",
        "project",
        "metric",
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


:call_me_hand: Community: [blue]{ea_constants.SLACK_URL}[/blue]

:e-mail: Email: [blue]{ea_constants.ENCORD_EMAIL}[/blue]

:star: Github: [blue]{ea_constants.GITHUB_URL}[/blue]
""",
)
cli.add_typer(config_cli, name="config", help="[green bold]Configure[/green bold] global settings 🔧")
cli.add_typer(import_cli, name="import", help="[green bold]Import[/green bold] projects or predictions ⬇️")
cli.add_typer(print_cli, name="print")
cli.add_typer(metric_cli, name="metric", help="[green bold]Manage[/green bold] project metrics :clipboard:")
cli.add_typer(project_cli, name="project", help="[green bold]Manage[/green bold] project settings ⚙️")


@cli.command()
def download(
    project_name: str = typer.Option(None, help="Name of the chosen project."),
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Directory where the project would be saved.", file_okay=False
    ),
):
    """
    [green bold]Download[/green bold] a sandbox dataset to get started 📁

    * If the argument --project-name is not provided, a list of available sandbox projects will be displayed, allowing you to select one from the menu.
    """
    from InquirerPy import inquirer as i

    from encord_active.lib.project.sandbox_projects import (
        available_prebuilt_projects,
        fetch_prebuilt_project_size,
    )

    if project_name is not None and project_name not in available_prebuilt_projects():
        rich.print("No such project in prebuilt projects.")
        raise typer.Abort()

    if not project_name:
        rich.print("Loading prebuilt projects ...")
        project_names_with_storage = []
        downloaded = {fetch_project_meta(project_path)["project_hash"] for project_path in find_child_projects(target)}
        for project_name, data in available_prebuilt_projects().items():
            if data["hash"] in downloaded:
                continue
            project_size = fetch_prebuilt_project_size(project_name)
            modified_project_name = project_name + (f" ({project_size} MB)" if project_size is not None else "")
            project_names_with_storage.append(modified_project_name)

        if not project_names_with_storage:
            rich.print("[green]Nothing to download, current working directory contains all sandbox projects.")
            raise typer.Exit()

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
    ensure_safe_project(project_path)
    success_with_vizualise_command(project_path, "Successfully downloaded sandbox dataset. ")


@cli.command(
    name="init",
)
def import_local_project(
    root: Path = typer.Argument(
        ...,
        help="The root directory of the dataset you are trying to import",
        file_okay=False,
    ),
    data_glob: List[str] = typer.Option(
        ["**/*.jpg", "**/*.png", "**/*.jpeg", "**/*.tiff"],
        "--data-glob",
        "-dg",
        help="Glob pattern to choose files. Repeat the `--data-glob` argument to match multiple globs.",
    ),
    label_glob: List[str] = typer.Option(
        None,
        "--label-glob",
        "-lg",
        help="Glob pattern to choose label files. Repeat the `--label-glob` argument to match multiple globs. This argument is only used if you also provide the `transformer` argument.",
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
    transformer: Path = typer.Option(
        None,
        help="Path to python module with one or more implementations of the `[blue]encord_active.lib.labels.label_transformer.LabelTransformer[/blue]` interface.",
        exists=True,
    ),
):
    """
    [green bold]Initialize[/green bold] a project from your local file system :seedling:

    The command will search for images based on the [blue]`data-glob`[/blue] arguments.
    By default, all jpeg, jpg, png, and tiff files will be matched.

    It will search for label files with the [blue]`label-glob`[/blue] argument.

    Both glob results will be passed to your implementation of the `LabelTransformer` interface if you provide a `transformer` argument.

    """
    from encord.ontology import OntologyStructure
    from InquirerPy import inquirer as i
    from InquirerPy.base.control import Choice

    from encord_active.lib.labels.label_transformer import (
        TransformerResult,
        load_transformers_from_module,
    )
    from encord_active.lib.metrics.execute import (
        run_metrics,
        run_metrics_by_embedding_type,
    )
    from encord_active.lib.metrics.heuristic.img_features import AreaMetric
    from encord_active.lib.metrics.types import EmbeddingType
    from encord_active.lib.project.local import (
        NoFilesFoundError,
        ProjectExistsError,
        file_glob,
        init_local_project,
    )
    from encord_active.lib.project.project_file_structure import ProjectFileStructure

    try:
        data_result = file_glob(root, data_glob, images_only=True)
    except NoFilesFoundError as e:
        rich.print(
            Panel(
                str(e),
                title=":fire: No files found from data glob :fire:",
                expand=False,
                style="yellow",
            )
        )
        raise typer.Abort()

    if not label_glob:
        label_result: List[Path] = []
    else:
        if transformer is None:
            rich.print("Label glob specified without a transformer. Label glob is only allowed with a transformer.")
            print(label_glob)
            raise typer.Abort()

        try:
            label_result = file_glob(root, label_glob, images_only=False).matched
        except NoFilesFoundError as e:
            rich.print(
                Panel(
                    str(e),
                    title=":fire: No files found from label glob :fire:",
                    expand=False,
                    style="yellow",
                )
            )
            raise typer.Abort()

    selected_transformer: Optional[TransformerResult] = None
    if transformer is not None:
        try:
            transformers_found = load_transformers_from_module(transformer)
        except (ModuleLoadError, ValueError) as e:
            rich.print(e)
            raise typer.Abort()

        if not transformers_found:
            rich.print(f"[yellow]Didn't find any transformers in `[blue]{transformer}[/blue]`")
            raise typer.Abort()
        elif len(transformers_found) == 1:
            selected_transformer = transformers_found[0]
        else:
            choices = list(map(lambda m: Choice(m, name=m.name), transformers_found))
            selected_transformer = i.select(
                message="Please choose which label transformer to use? Use [ENTER] to select from the list.",
                choices=choices,
            ).execute()

    if dryrun:
        directories: Set[Path] = set()
        rich.print("[blue]Included files:[/blue]")
        for file in data_result.matched:
            directories.add(file.parent)
            rich.print(f"[blue]{escape(file.as_posix())}[/blue]")

        print()
        if data_result.excluded:
            rich.print("[yellow]Excluded files:[/yellow]")
            for file in data_result.excluded:
                rich.print(f"[yellow]{escape(file.as_posix())}[/yellow]")

        label_stats = ""
        total_labels = 0
        if selected_transformer is not None:
            labels = selected_transformer.transformer.from_custom_labels(label_result, data_files=data_result.matched)
            labels = sorted(labels, key=lambda l: l.abs_data_path)

            if not labels:
                rich.print(f"[yellow]The transformer [blue]{selected_transformer.name}[/blue] didn't return any labels")
            else:
                rich.print(f"Labels identified by [blue]{selected_transformer.name}[/blue]:")

                found_labels: Dict[str, Dict[str, int]] = {}  # <type, <class, label>>
                current_file_name = None
                for label in labels:
                    label_type = type(label.label).__name__
                    label_name = label.label.class_
                    counter = found_labels.setdefault(label_type, {})
                    counter[label_name] = counter.get(label_name, 0) + 1

                    if label.abs_data_path != current_file_name:
                        current_file_name = label.abs_data_path
                        rich.print(f"[green]{current_file_name}[/green]")
                    rich.print(f"\t{label.label}")

                total_labels = sum([sum(v.values()) for v in found_labels.values()])
                for label_type, counts in found_labels.items():
                    label_stats += f"\t{label_type}\n"
                    label_stats += "\n".join([f"\t\t{k}: {v}" for k, v in counts.items()])
                    label_stats += "\n"

        exclusion = ""
        if len(data_result.excluded):
            exclusion = f"[yellow]Excluded[/yellow] {len(data_result.excluded)} file(s) because they do not seem to be images.\n"

        print()
        rich.print(
            Panel(
                f"""
[blue]Found[/blue] {len(data_result.matched)} file(s) in {len(directories)} directories.
{exclusion}
[blue]Found[/blue] {total_labels} label(s):
{label_stats}
""",
                title=":bar_chart: Stats :bar_chart:",
                expand=False,
            )
        )

        raise typer.Exit()

    transformer_instance = selected_transformer.transformer if selected_transformer else None

    if not project_name:
        project_name = f"[EA] {root.name}"

    try:
        project_path = init_local_project(
            files=data_result.matched,
            target=target,
            project_name=project_name,
            symlinks=symlinks,
            label_transformer=transformer_instance,
            label_paths=label_result,
        )

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

    metricize_options = {"data_dir": project_path, "use_cache_only": True}
    if metrics:
        run_metrics_by_embedding_type(EmbeddingType.IMAGE, **metricize_options)

        ontology = OntologyStructure.from_dict(json.loads(ProjectFileStructure(project_path).ontology.read_text()))
        if ontology.objects:
            run_metrics_by_embedding_type(EmbeddingType.OBJECT, **metricize_options)
        if ontology.classifications:
            run_metrics_by_embedding_type(EmbeddingType.CLASSIFICATION, **metricize_options)

    else:
        # NOTE: we need to compute at least one metric otherwise everything breaks
        run_metrics(filter_func=lambda x: isinstance(x, AreaMetric), **metricize_options)

    success_with_vizualise_command(project_path, "Project initialised :+1:")


@cli.command(name="refresh")
@ensure_project()
def refresh(
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False)
):
    """
    [green bold]Sync[/green bold] data and labels from a remote Encord project :arrows_counterclockwise:

    The local project should have a reference to the remote Encord project in its config file (`project_meta.yaml`).
    The required attributes are:
    1. The remote flag set to `true` (has_remote: true).
    2. The hash of the remote Encord project (project_hash: remote-encord-project-hash).
    3. The path to the private Encord user SSH key (ssh_key_path: private/encord/user/ssh/key/path).
    """
    from encord_active.lib.project import Project

    try:
        Project(target).refresh()
    except Exception as e:
        rich.print(f"[red] ERROR: The data sync failed. Log: {e}.")
    else:
        rich.print("[green]Data and labels successfully synced from the remote project[/green]")


@cli.command(name="start")
def start(
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Path of the project you would like to start", file_okay=False
    ),
):
    """
    [green bold]Launch[/green bold] the application with the provided project ✨
    """
    from encord_active.cli.utils.server import launch_server_app

    launch_server_app(target)


@cli.command()
def quickstart(
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Directory where the project would be saved.", file_okay=False
    ),
):
    """
    [green bold]Start[/green bold] Encord Active straight away 🏃💨
    """
    from encord_active.cli.utils.server import launch_server_app
    from encord_active.lib.project.sandbox_projects import fetch_prebuilt_project

    project_name = "quickstart"
    project_dir = target / project_name
    project_dir.mkdir(exist_ok=True)

    fetch_prebuilt_project(project_name, project_dir)
    launch_server_app(project_dir)


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


@cli.command(name="join-discord", rich_help_panel="Resources")
def join_discord():
    """
    [green bold]Join[/green bold] the Discord family :family:
    While our primary community hub is on Slack, you're welcome to connect with us on Discord as well.
    """
    import webbrowser

    webbrowser.open(ea_constants.JOIN_DISCORD_URL)


@cli.callback(invoke_without_command=True)
def version(version_: bool = typer.Option(False, "--version", "-v", help="Print the current version of Encord Active")):
    if version_:
        from encord_active import __version__ as ea_version

        rich.print(f"Version: [green]{ea_version}[/green]")
        exit()


if __name__ == "__main__":
    cli(prog_name=APP_NAME)
