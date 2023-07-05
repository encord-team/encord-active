from pathlib import Path
from typing import Optional, TypedDict

import rich
import typer
from loguru import logger
from rich.table import Table, box

from encord_active.cli.utils.decorators import ensure_project
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics, build_merged_metrics
from encord_active.lib.metrics.metadata import fetch_metrics_meta, update_metrics_meta
from encord_active.lib.project import ProjectFileStructure

metric_cli = typer.Typer(rich_markup_mode="markdown")
logger = logger.opt(colors=True)
SIMPLE_HEAD_BOX = box.Box("    \n    \n -  \n    \n    \n    \n    \n    \n")


@metric_cli.command(name="add", short_help="Add metrics.")
@ensure_project()
def add_metrics(
    module_path: Path = typer.Argument(
        ..., help="Path to the python module where the metric resides.", exists=True, dir_okay=False
    ),
    metric_title: Optional[list[str]] = typer.Argument(None, help="Title of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """
    Add metrics to the project.

    Add metrics from:
    - local python modules

    If no metric titles are provided, all metrics found in the Python module will be automatically added to the project.
    """
    rich.print("Inspecting provided module ...")

    from encord_active.lib.metrics.io import (
        get_metric_metadata,
        get_metrics,
        get_module_metrics,
    )

    full_module_path = module_path.expanduser().resolve()
    metric_titles = dict.fromkeys(metric_title or [], full_module_path)

    project_file_structure = ProjectFileStructure(target)
    metrics_meta = fetch_metrics_meta(project_file_structure)

    add_all_metrics = len(metric_titles) == 0
    if add_all_metrics:
        # extract all metrics from the module as no metric title was provided
        found_metrics = get_module_metrics(full_module_path, lambda x: True)
    else:
        # faster extraction of selected metrics from the module
        found_metrics = get_metrics(list(metric_titles.items()))
    if found_metrics is None:
        rich.print("[red]Error: Provided module path doesn't comply with expected format. Check the logs.[/red]")
        raise typer.Abort()

    found_metric_titles = {metric.metadata.title: metric for metric in found_metrics}
    if add_all_metrics:
        metric_titles = dict.fromkeys(found_metric_titles.keys(), full_module_path)

    has_conflicts = False
    for title in metric_titles:
        if title in found_metric_titles:
            # input metric title was found in the python module
            if title in metrics_meta:
                old_module_path = metrics_meta[title]["location"]
                if full_module_path == Path(old_module_path).expanduser().resolve():
                    rich.print(f"Requirement already satisfied: {title} in '{module_path.as_posix()}'.")
                else:
                    rich.print(
                        f"[red]ERROR: Conflict found in metric {title}. "
                        f"The same metric title already points to '{old_module_path}'. "
                        "Please change the metric title if you intend to have both metrics included.[/red]"
                    )
                    has_conflicts = True
            else:
                # attach input metric to the project and recreate its metadata in metrics_meta.json
                metrics_meta[title] = get_metric_metadata(found_metric_titles[title], module_path)
                rich.print(f"Adding {title}")
        else:
            # input metric title was not found in the python module
            rich.print(f"[red]ERROR: No matching metric found for {title}.[/red]")
            has_conflicts = True

    # update metric dependencies in metrics_meta.json
    update_metrics_meta(project_file_structure, metrics_meta)
    if has_conflicts:
        rich.print("[yellow]Errors were found. Not all metrics were successfully added.[/yellow]")
    else:
        rich.print("[green]Successfully added all metrics.[/green]")


@metric_cli.command(name="list", short_help="List metrics.")
@ensure_project()
def list_metrics(
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """
    List metrics in the project, including editables.

    Metrics are listed in a case-insensitive sorted order.
    """
    metrics_meta = fetch_metrics_meta(ProjectFileStructure(target))
    sorted_titles = sorted(metrics_meta.keys(), key=lambda x: x.lower())

    table = Table(box=SIMPLE_HEAD_BOX, show_edge=False, padding=0)
    table.add_column("Metric Title")
    table.add_column("Editable metric location")
    for title in sorted_titles:
        table.add_row(title, metrics_meta[title]["location"])
    rich.print(table)


@metric_cli.command(name="remove", short_help="Remove metrics.")
@ensure_project()
def remove_metrics(
    metric_title: list[str] = typer.Argument(..., help="Title of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """Remove metrics from the project."""
    project_file_structure = ProjectFileStructure(target)
    metrics_meta = fetch_metrics_meta(project_file_structure)

    for title in metric_title:
        if title in metrics_meta:
            rich.print(f"Found existing metric: {title}")
            rich.print(f"Removing {title}:")
            rich.print("  Would detach:")
            rich.print(f"    {metrics_meta[title]['location']}")
            ok_remove = typer.confirm("Proceed?")
            if ok_remove:
                metrics_meta.pop(title)
                rich.print(f"Successfully removed {title}")
        else:
            rich.print(f"[yellow]WARNING: Skipping {title} as it is not attached to the project.[/yellow]")

    # update metric dependencies in metrics_meta.json
    update_metrics_meta(project_file_structure, metrics_meta)


@metric_cli.command(name="run", short_help="Run metrics.")
@ensure_project()
def run_metrics(
    metric_title: Optional[list[str]] = typer.Argument(None, help="Title of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
    run_all: bool = typer.Option(False, "--all", help="Run all metrics."),
    fuzzy: bool = typer.Option(
        False, help="Enable fuzzy search in the selection. (press [TAB] or [SPACE] to select more than one) 🪄"
    ),
):
    """Run metrics on project data and labels."""
    from InquirerPy import inquirer as i
    from InquirerPy.base.control import Choice

    from encord_active.lib.metrics.execute import execute_metrics
    from encord_active.lib.metrics.io import get_metrics

    project_file_structure = ProjectFileStructure(target)
    metrics_meta = fetch_metrics_meta(project_file_structure)
    metrics = get_metrics([(m_title, m_meta["location"]) for m_title, m_meta in metrics_meta.items()])
    if run_all:  # User chooses to run all available metrics
        selected_metrics = metrics

    # (interactive) User chooses some metrics via CLI prompt selection
    elif not metric_title:
        choices = list(map(lambda m: Choice(m, name=m.metadata.title), metrics))
        Options = TypedDict("Options", {"message": str, "choices": list[Choice], "vi_mode": bool})
        options: Options = {
            "message": "What metrics would you like to run?\n> Press [TAB] or [SPACE] to select more than one.",
            "choices": choices,
            "vi_mode": True,
        }

        if fuzzy:
            selected_metrics = i.fuzzy(**options, multiselect=True).execute()
        else:
            selected_metrics = i.checkbox(**options).execute()

    # (non-interactive) User chooses some metrics via CLI argument
    else:
        metric_title_to_cls = {m.metadata.title: m for m in metrics}
        used_metric_titles = set()
        selected_metrics = []
        unknown_metric_titles = []
        for title in metric_title:
            if title in used_metric_titles:  # ignore repeated metric titles in the CLI argument
                continue
            used_metric_titles.add(title)

            metric_cls = metric_title_to_cls.get(title, None)
            if metric_cls is None:  # unknown and/or wrong metric title in user selection
                unknown_metric_titles.append(title)
            else:
                selected_metrics.append(metric_cls)

        if len(unknown_metric_titles) > 0:
            rich.print("No available metric with this title:")
            for title in unknown_metric_titles:
                rich.print(f"[yellow]{title}[/yellow]")
            raise typer.Abort()

    execute_metrics(selected_metrics, data_dir=target, use_cache_only=True)
    with DBConnection(project_file_structure) as conn:
        MergedMetrics(conn).replace_all(build_merged_metrics(project_file_structure.metrics))


@metric_cli.command(name="show", short_help="Show information about available metrics.")
@ensure_project()
def show_metrics(
    metric_title: list[str] = typer.Argument(..., help="Title of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """Show information about one or more available metrics in the project."""

    metrics_meta = fetch_metrics_meta(ProjectFileStructure(target))

    not_found_metrics = [title for title in metric_title if title not in metrics_meta]
    if len(not_found_metrics) > 0:
        rich.print("[yellow]WARNING: Package(s) not found:", ", ".join(not_found_metrics))

    first = True
    hidden_properties = {"stats", "long_description"}
    for title in metric_title:
        if title in metrics_meta:
            if not first:
                print("---")
            print(
                *(
                    f"{_property.replace('_', ' ').capitalize()}: {value}"
                    for _property, value in metrics_meta[title].items()
                    if _property not in hidden_properties
                ),
                sep="\n",
            )
            first = False
