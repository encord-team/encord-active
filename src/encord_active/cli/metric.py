from pathlib import Path
from typing import Optional, TypedDict

import rich
import typer
from loguru import logger
from rich.table import Table, box

from encord_active.cli.utils.decorators import ensure_project
from encord_active.lib.project.metadata import fetch_project_meta, update_project_meta

metric_cli = typer.Typer(rich_markup_mode="markdown")
logger = logger.opt(colors=True)
SIMPLE_HEAD_BOX = box.Box("    \n    \n -  \n    \n    \n    \n    \n    \n")


@metric_cli.command(name="add", short_help="Add metrics.")
@ensure_project
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

    If no metric title is provided then all metrics found in the python module will be added to the project.
    """
    rich.print("Inspecting provided module ...")

    from encord_active.lib.metrics.io import get_metrics, get_module_metrics

    full_module_path = module_path.expanduser().resolve()
    metric_titles = dict.fromkeys(metric_title or [], full_module_path)

    project_meta = fetch_project_meta(target)
    # dict.setdefault ensures backwards compatibility on projects without registered metrics
    project_metrics = project_meta.setdefault("metrics", dict())

    add_all_metrics = len(metric_titles) == 0
    if add_all_metrics:
        # extract all metrics from the module as no metric title was provided
        found_metrics = get_module_metrics(full_module_path, lambda x: True)
    else:
        # faster extraction of selected metrics from the module
        found_metrics = get_metrics(list(metric_titles.items()))
    if found_metrics is None:
        rich.print("[red]Error: Provided module path doesn't comply with expected format. Check the logs.[/red]")
        return

    found_metric_titles = dict.fromkeys((metric.metadata.title for metric in found_metrics), full_module_path)
    if add_all_metrics:
        metric_titles = found_metric_titles.copy()

    has_conflicts = False
    for title in metric_titles:
        if title in found_metric_titles:
            # input metric title was found in the python module
            if title in project_metrics:
                old_module_path = project_metrics[title]
                if full_module_path == Path(old_module_path).expanduser().resolve():
                    rich.print(f"Requirement already satisfied: {title} in '{module_path.as_posix()}'.")
                else:
                    rich.print(f"[red]ERROR: Conflict found in metric {title}. It points to '{old_module_path}'.[/red]")
                    has_conflicts = True
            else:
                project_metrics[title] = module_path.as_posix()
                rich.print(f"Adding {title}")
        else:
            # input metric title was not found in the python module
            rich.print(f"[red]ERROR: No matching metric found for {title}.[/red]")
            has_conflicts = True

    # update metric dependencies in project_meta.yaml
    update_project_meta(target, project_meta)
    if has_conflicts:
        rich.print("[yellow]Errors were found. Not all metrics were successfully added.[/yellow]")
    else:
        rich.print("[green]Successfully added all metrics.[/green]")


@metric_cli.command(name="list", short_help="List metrics.")
@ensure_project
def list_metrics(
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """
    List metrics in the project, including editables.

    Metrics are listed in a case-insensitive sorted order.
    """
    project_meta = fetch_project_meta(target)
    # dict.setdefault ensures backwards compatibility on projects without registered metrics
    project_metrics = project_meta.setdefault("metrics", dict())
    sorted_titles = sorted(project_metrics.keys(), key=lambda x: x.lower())

    table = Table(box=SIMPLE_HEAD_BOX, show_edge=False, padding=0)
    table.add_column("Metric Title")
    table.add_column("Editable metric location")
    for title in sorted_titles:
        table.add_row(title, project_metrics[title])
    rich.print(table)


@metric_cli.command(name="remove", short_help="Remove metrics.")
@ensure_project
def remove_metrics(
    metric_title: list[str] = typer.Argument(..., help="Title of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """Remove metrics from the project."""
    project_meta = fetch_project_meta(target)
    # dict.setdefault ensures backwards compatibility on projects without registered metrics
    project_metrics = project_meta.setdefault("metrics", dict())

    for title in metric_title:
        if title in project_metrics:
            rich.print(f"Found existing metric: {title}")
            rich.print(f"Removing {title}:")
            rich.print("  Would detach:")
            rich.print(f"    {project_metrics[title]}")
            ok_remove = typer.confirm("Proceed?")
            if ok_remove:
                project_metrics.pop(title)
                rich.print(f"Successfully removed {title}")
        else:
            rich.print(f"[yellow]WARNING: Skipping {title} as it is not attached to the project.[/yellow]")

    # update metric dependencies in project_meta.yaml
    update_project_meta(target, project_meta)


@metric_cli.command(name="run", short_help="Run metrics.")
@ensure_project
def run_metrics(
    metric_title: Optional[list[str]] = typer.Argument(None, help="Title of the metric. Can be used multiple times."),
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
    # dict.setdefault ensures backwards compatibility on projects without registered metrics
    project_metrics = project_meta.setdefault("metrics", dict())

    metrics = get_metrics(list(project_metrics.items()))
    if run_all:  # User chooses to run all available metrics
        selected_metrics = metrics

    # (interactive) User chooses some metrics via CLI prompt selection
    elif not metric_title:
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


@metric_cli.command(name="show", short_help="Show information about available metrics.")
@ensure_project
def show_metrics(
    metric_title: list[str] = typer.Argument(..., help="Title of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """Show information about one or more available metrics in the project."""
    pass
