import pickle
from pathlib import Path
from typing import Optional

import rich
import typer
from rich.markup import escape
from rich.panel import Panel

from encord_active.cli.config import app_config
from encord_active.cli.utils.decorators import ensure_project

import_cli = typer.Typer(rich_markup_mode="markdown")


@import_cli.command(name="predictions")
@ensure_project
def import_predictions(
    predictions_path: Path = typer.Argument(
        ...,
        help="Path to a predictions file.",
        dir_okay=False,
    ),
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
    coco: bool = typer.Option(False, help="Import a coco result format file"),
):
    """
    [bold]Imports[/bold] a predictions file. The predictions should be using the `Prediction` model and be stored in a pkl file.
    If `--coco` option is specified the file should be a json following the coco results format. :brain:
    """
    from encord_active.lib.project import Project

    project = Project(target)

    if coco:
        from encord_active.cli.utils.coco import import_coco_predictions

        predictions = import_coco_predictions(target, predictions_path)
    else:
        with open(predictions_path, "rb") as f:
            predictions = pickle.load(f)

    from encord_active.lib.db.predictions import (
        import_predictions as app_import_predictions,
    )

    app_import_predictions(project, target, predictions)


ENCORD_RICH_PANEL = "Encord Project Arguments"
COCO_RICH_PANEL = "COCO Project Arguments"
REQUIRED_WITH_COCO = f"[dark_red]{escape('[required with --coco]')}[/dark_red]"


@import_cli.command(name="project")
def import_project(
    target: Path = typer.Option(
        Path.cwd(),
        "--target",
        "-t",
        help="Directory where the project would be saved.",
        file_okay=False,
    ),
    # Encord
    encord_project_hash: Optional[str] = typer.Option(
        None,
        "--project_hash",
        help="Encord project hash of the project you wish to import. Leaving it blank will allow you to choose one interactively.",
        rich_help_panel=ENCORD_RICH_PANEL,
    ),
    # COCO
    coco: bool = typer.Option(False, help="Import a project from the coco format", rich_help_panel=COCO_RICH_PANEL),
    images: Optional[Path] = typer.Option(
        None,
        "--images",
        "-i",
        help=f"Path to the directory containing the dataset images. {REQUIRED_WITH_COCO}",
        file_okay=False,
        exists=True,
        rich_help_panel=COCO_RICH_PANEL,
    ),
    annotations: Optional[Path] = typer.Option(
        None,
        "--annotations",
        "-a",
        help=f"Path to the file containing the dataset annotations. {REQUIRED_WITH_COCO}",
        dir_okay=False,
        exists=True,
        rich_help_panel=COCO_RICH_PANEL,
    ),
    symlinks: bool = typer.Option(
        False,
        help="Use symlinks instead of copying COCO images to the target directory.",
        rich_help_panel=COCO_RICH_PANEL,
    ),
):
    """
    [bold]Imports[/bold] a new project from Encord or a local coco project 📦

    Confirm with each help panel what information you will have to provide.
    """
    if coco:
        from encord_active.cli.utils.coco import import_coco_project

        if not annotations:
            raise typer.BadParameter("`annotations` argument is missing")
        elif not images:
            raise typer.BadParameter("`images` argument is missing")

        project_path = import_coco_project(images, annotations, target, use_symlinks=symlinks)
    else:
        from encord_active.cli.utils.encord import import_encord_project

        ssh_key_path = app_config.get_or_query_ssh_key()
        project_path = import_encord_project(ssh_key_path, target, encord_project_hash)

    cwd = Path.cwd()
    cd_dir = project_path.relative_to(cwd) if project_path.is_relative_to(cwd) else project_path

    panel = Panel(
        f"""
    The data is downloaded and the metrics are complete.

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
