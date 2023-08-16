from pathlib import Path
from typing import Optional

import rich
import typer
from loguru import logger
from rich.table import Table, box

from encord_active.cli.common import (
    TYPER_ENCORD_DATABASE_DIR,
    TYPER_SELECT_PROJECT_NAME,
    select_project_hash_from_name,
)

metric_cli = typer.Typer(rich_markup_mode="markdown")
logger = logger.opt(colors=True)
SIMPLE_HEAD_BOX = box.Box("    \n    \n -  \n    \n    \n    \n    \n    \n")

'''

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
'''


@metric_cli.command(name="list", short_help="List metrics.")
def list_metrics(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    project_name: Optional[str] = TYPER_SELECT_PROJECT_NAME,
) -> None:
    """
    List metrics in the project, including editables.

    Metrics are listed in a case-insensitive sorted order.
    """
    from encord_active.db.metrics import AnnotationMetrics, DataMetrics

    data_metrics = DataMetrics  # FIXME: change to custom metric supporting lookup
    annotation_metrics = AnnotationMetrics
    sorted_metrics = sorted((data_metrics | annotation_metrics).items(), key=lambda x: x[1].title)

    table = Table(box=SIMPLE_HEAD_BOX, show_edge=False, padding=0)
    table.add_column("Metric Title")
    table.add_column("Data")
    table.add_column("Annotation")
    table.add_column("Custom")
    for metric_key, metric in sorted_metrics:
        _bool = {
            True: rich.table.Text("Y", style="GREEN"),
            False: rich.table.Text("N", style="RED"),
        }
        table.add_row(
            metric.title,
            _bool[metric_key in data_metrics],
            _bool[metric_key in annotation_metrics],
            _bool[False],
        )
    rich.print(table)


'''
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
'''


@metric_cli.command(name="run", short_help="Run metrics.")
def run_metrics(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    project_name: Optional[str] = TYPER_SELECT_PROJECT_NAME,
):
    """Run metrics on project data and labels."""
    raise ValueError("Not yet implemented")


@metric_cli.command(name="show", short_help="Show information about available metrics.")
def show_metrics(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    project_name: Optional[str] = TYPER_SELECT_PROJECT_NAME,
) -> None:
    """Show information about one or more available metrics in the project."""
    from sqlmodel import Session

    from encord_active.db.models import get_engine
    from encord_active.server.routers.queries.domain_query import (
        TABLES_ANNOTATION,
        TABLES_DATA,
    )
    from encord_active.server.routers.queries.metric_query import query_attr_summary

    project_hash = select_project_hash_from_name(database_dir, project_name or "")

    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        for table_name, tables in [("Data", TABLES_DATA), ("Annotation", TABLES_ANNOTATION)]:
            summary = query_attr_summary(
                sess=sess,
                tables=tables,
                project_filters={"project_hash": [project_hash]},
                filters=None,
            )
            metrics = (tables.annotation or tables.data).metrics
            summary_metrics = list(summary.metrics.items())
            summary_metrics.sort(key=lambda x: metrics[x[0]].title)
            table = Table(box=SIMPLE_HEAD_BOX, show_edge=False, padding=0, title=f"{table_name} Metric Summary")
            table.add_column("Metric Title")
            table.add_column("Min")
            table.add_column("Q1")
            table.add_column("Median")
            table.add_column("Q3")
            table.add_column("Max")
            table.add_column("Moderate Outlier")
            table.add_column("Severe Outlier")
            table.add_column("Samples")
            for metric_key, metric_summary in summary_metrics:
                if metric_summary.count > 0:
                    table.add_row(
                        metrics[metric_key].title,
                        str(metric_summary.min),
                        str(metric_summary.q1),
                        str(metric_summary.median),
                        str(metric_summary.q3),
                        str(metric_summary.max),
                        str(metric_summary.moderate),
                        str(metric_summary.severe),
                        str(metric_summary.count),
                    )
            rich.print(table)
