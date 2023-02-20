from pathlib import Path
from typing import Optional

import typer

from encord_active.cli.utils.decorators import ensure_project

metric_cli = typer.Typer(rich_markup_mode="markdown")


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
    pass


@metric_cli.command(name="list", short_help="List available metrics.")
@ensure_project
def list_metrics(
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """
    List available metrics in the project, including editables.

    Metrics are listed in a case-insensitive sorted order.
    """
    pass


@metric_cli.command(name="remove", short_help="Remove metrics.")
@ensure_project
def remove_metrics(
    metric_name: list[str] = typer.Argument(..., help="Name of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """Remove metrics from the project."""
    pass


@metric_cli.command(name="show", short_help="Show information about available metrics.")
@ensure_project
def show_metrics(
    metric_name: list[str] = typer.Argument(..., help="Name of the metric. Can be used multiple times."),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
):
    """Show information about one or more available metrics in the project."""
    pass
