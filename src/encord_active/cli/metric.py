import importlib.util
import inspect
from pathlib import Path
from typing import Callable, Optional, TypedDict, Union

import rich
import typer
from loguru import logger
from rich.table import Table, box

from encord_active.cli.utils.decorators import ensure_project
from encord_active.lib.metrics.io import get_module_metrics
from encord_active.lib.metrics.metric import Metric, SimpleMetric
from encord_active.lib.project.metadata import fetch_project_meta, update_project_meta

metric_cli = typer.Typer(rich_markup_mode="markdown")
logger = logger.opt(colors=True)
SIMPLE_HEAD_BOX = box.Box("    \n    \n -  \n    \n    \n    \n    \n    \n")


@metric_cli.command(name="add", short_help="Add metrics.")
@ensure_project
def add_metrics(
    metric_path: Path = typer.Argument(
        ..., help="Path to the python module where the metric resides.", exists=True, dir_okay=False
    ),
    metric_name: Optional[list[str]] = typer.Argument(None, help="Name of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """
    Add metrics to the project.

    Add metrics from:
    - local source python modules

    If no metric name is provided then all metrics found in the python module will be added to the project.
    """
    rich.print("Inspecting provided module ...")
    metric_names = [] if metric_name is None else metric_name
    metric_names_set = set(metric_names)
    expanded_metric_path = metric_path.expanduser().resolve()

    project_meta = fetch_project_meta(target)
    project_metrics = project_meta.setdefault("metrics", dict())  # ensure backwards compatibility

    found_metrics = get_module_metrics(
        expanded_metric_path, lambda x: len(metric_names) == 0 or x.__name__ in metric_names_set
    )
    if found_metrics is None:
        rich.print("[red]Error: Provided module path doesn't comply with expected format. Check the logs.[/red]")
        return
    found_metrics_set = set(found_metrics)

    # add all metrics from the python module if the input doesn't contain any metric name.
    if len(metric_names) == 0:
        metric_names = found_metrics.copy()
        metric_names_set = found_metrics_set.copy()

    has_conflicts = False
    for m_name in metric_names:
        # discard repeated occurrences of an already processed metric
        if m_name not in metric_names_set:
            continue
        metric_names_set.discard(m_name)

        if m_name in found_metrics_set:
            # input metric name was found in the python module
            if m_name in project_metrics:
                old_metric_path = project_metrics[m_name]
                if expanded_metric_path == Path(old_metric_path).expanduser().resolve():
                    rich.print(f"Requirement already satisfied: {m_name} in '{metric_path.as_posix()}'")
                else:
                    rich.print(f"[red]ERROR: Conflict found in metric {m_name}. It points to {old_metric_path}'[/red]")
                    has_conflicts = True
            else:
                project_metrics[m_name] = metric_path.as_posix()
                rich.print(f"Adding {m_name}")
        else:
            # input metric name was not found in the python module
            rich.print(f"[red]ERROR: No matching metric found for {m_name} [/red]")
            has_conflicts = True

    # update metric dependencies in project_meta.yaml
    update_project_meta(target, project_meta)
    if has_conflicts:
        rich.print("[yellow]Errors were found. Not all metrics were successfully added.[/yellow]")
    else:
        rich.print("[green]Successfully added all metrics.[/green]")


@metric_cli.command(name="list", short_help="List available metrics.")
@ensure_project
def list_metrics(
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """
    List available metrics in the project, including editables.

    Metrics are listed in a case-insensitive sorted order.
    """
    project_meta = fetch_project_meta(target)
    project_metrics = project_meta.setdefault("metrics", dict())  # ensure backwards compatibility
    sorted_names = sorted(project_metrics.keys(), key=lambda x: x.lower())

    table = Table(box=SIMPLE_HEAD_BOX, show_edge=False, padding=0)
    table.add_column("Metric")
    table.add_column("Editable metric location")
    for m_name in sorted_names:
        table.add_row(m_name, project_metrics[m_name])
    rich.print(table)


@metric_cli.command(name="remove", short_help="Remove metrics.")
@ensure_project
def remove_metrics(
    metric_name: list[str] = typer.Argument(..., help="Name of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """Remove metrics from the project."""
    project_meta = fetch_project_meta(target)
    project_metrics = project_meta.setdefault("metrics", dict())  # ensure backwards compatibility

    for m_name in metric_name:
        if m_name in project_metrics:
            rich.print(f"Found existing metric: {m_name}")
            rich.print(f"Removing {m_name}:")
            rich.print(f"  Would detach:")
            rich.print(f"    {project_metrics[m_name]}")
            ok_remove = typer.confirm("Proceed?")
            if ok_remove:
                project_metrics.pop(m_name)
                rich.print(f"Successfully removed {m_name}")
        else:
            rich.print(f"[yellow]WARNING: Skipping {m_name} as it is not attached to the project.[/yellow]")

    # update metric dependencies in project_meta.yaml
    update_project_meta(target, project_meta)


@metric_cli.command(name="run", short_help="Run metrics.")
@ensure_project
def run_metrics(
    metric_name: Optional[list[str]] = typer.Argument(None, help="Name of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
    run_all: bool = typer.Option(False, "--all", help="Run all metrics."),
    fuzzy: bool = typer.Option(
        False, help="Enable fuzzy search in the selection. (press [TAB] or [SPACE] to select more than one) 🪄"
    ),
):
    """Run metrics on the project's data."""
    from InquirerPy import inquirer as i
    from InquirerPy.base.control import Choice

    from encord_active.lib.metrics.execute import execute_metrics
    from encord_active.lib.metrics.io import get_metrics

    project_meta = fetch_project_meta(target)
    project_metrics = project_meta.setdefault("metrics", dict())  # ensure backwards compatibility

    metrics = get_metrics(project_metrics.items())
    if run_all:  # User chooses to run all available metrics
        selected_metrics = metrics

    # (interactive) User chooses some metrics via CLI prompt selection
    elif not metric_name:
        choices = list(map(lambda m: Choice(m, name=m.metadata.title), metrics))
        Options = TypedDict("Options", {"message": str, "choices": list[Choice], "vi_mode": bool})
        options: Options = {
            "message": "What metrics would you like to run?",
            "choices": choices,
            "vi_mode": True,
        }

        if fuzzy:
            options["message"] += " [blue](press [TAB] or [SPACE] to select more than one)[/blue]"
            selected_metrics = i.fuzzy(**options, multiselect=True).execute()
        else:
            selected_metrics = i.checkbox(**options).execute()

    # (non-interactive) User chooses some metrics via CLI argument
    else:
        metric_name_to_cls = {m.metadata.title: m for m in metrics}
        used_metric_names = set()
        selected_metrics = []
        unknown_metric_names = []
        for name in metric_name:
            if name in used_metric_names:  # ignore repeated metric names in the CLI argument
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


@metric_cli.command(name="show", short_help="Show information about available metrics.")
@ensure_project
def show_metrics(
    metric_name: list[str] = typer.Argument(..., help="Name of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """Show information about one or more available metrics in the project."""
    pass
