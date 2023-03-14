import json
from pathlib import Path
from typing import Dict, List, Optional, Set, TypedDict

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
from encord_active.cli.metric import metric_cli
from encord_active.cli.print import print_cli
from encord_active.cli.utils.decorators import bypass_streamlit_question, ensure_project
from encord_active.cli.utils.prints import success_with_visualise_command
from encord_active.lib import constants as ea_constants


class OrderedPanelGroup(TyperGroup):
    COMMAND_ORDER = [
        "quickstart",
        "download",
        "init",
        "import",
        "visualize",
        "metric",
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
cli.add_typer(metric_cli, name="metric", help="[green bold]Manage[/green bold] project's metrics.")


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
        help="Glob pattern to choose label files. Repeat the `--label-glob` argument to match multiple globs. This argument is only used if you also provide the `transformer` argument",
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
        help="Path to python module with one or more implementations of the `[blue]encord_active.lib.labels.label_transformer.LabelTransformer[/blue]` interface",
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
    from encord_active.lib.metrics.metric import EmbeddingType
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
                title=":fire: No Files Found From Data Glob :fire:",
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
                    title=":fire: No Files Found From Label Glob :fire:",
                    expand=False,
                    style="yellow",
                )
            )
            raise typer.Abort()

    transformer_results: List[TransformerResult] = []
    if transformer is not None:
        transformers_found = load_transformers_from_module(transformer)
        if not transformers_found:
            pass
        elif len(transformers_found) == 1:
            transformer_results = transformers_found
        else:
            choices = list(map(lambda m: Choice(m, name=m.name), transformers_found))
            transformer_results = i.checkbox(
                message="Please choose which label adapters to use? Use [TAB] to select from the list.", choices=choices
            ).execute()

    if dryrun:
        directories: Set[Path] = set()
        rich.print("[blue]Matches:[/blue]")
        for file in data_result.matched:
            directories.add(file.parent)
            rich.print(f"[blue]{escape(file.as_posix())}[/blue]")

        print()
        rich.print("[yellow]Excluded:[/yellow]")
        for file in data_result.excluded:
            directories.add(file.parent)
            rich.print(f"[yellow]{escape(file.as_posix())}[/yellow]")

        found_labels: Dict[str, Dict[str, int]] = {}  # <type, <class, label>>
        for transformer_result in transformer_results:
            labels = transformer_result.transformer.from_custom_labels(label_result, data_files=data_result.matched)
            for label in labels:
                label_type = type(label.label).__name__
                label_name = label.label.class_
                counter = found_labels.setdefault(label_type, {})
                counter[label_name] = counter.get(label_name, 0) + 1

            if not labels:
                rich.print(f"[yellow]The transformer [blue]{transformer_result.name}[/blue] didn't return any labels")
                continue

            labels = sorted(labels, key=lambda l: l.abs_data_path)
            rich.print(f"Labels identified by [blue]{transformer_result.name}[/blue]:")
            current_file_name = None
            for label in labels:
                if label.abs_data_path != current_file_name:
                    current_file_name = label.abs_data_path
                    rich.print(f"[green]{current_file_name}[/green]")
                rich.print(f"\t{label.label}")

            print()

        total_labels = sum([sum(v.values()) for v in found_labels.values()])

        label_stats = ""
        for label_type, counts in found_labels.items():
            label_stats += f"\t{label_type}\n"
            label_stats += "\n".join([f"\t\t{k}: {v}" for k, v in counts.items()])
            label_stats += "\n"

        exclusion = "\n"
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

    transformers = list(map(lambda t: t.transformer, transformer_results))

    if not project_name:
        project_name = f"[EA] {root.name}"

    try:
        project_path = init_local_project(
            files=data_result.matched,
            target=target,
            project_name=project_name,
            symlinks=symlinks,
            label_transformers=transformers,
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

    success_with_visualise_command(project_path, "Project initialised :+1:")


@cli.command(name="visualise", hidden=True)  # Alias for backward compatibility
@cli.command(name="visualize")
@bypass_streamlit_question
@ensure_project(allow_multi=True)
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
@ensure_project()
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
