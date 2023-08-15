import pickle
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.markup import escape

from encord_active.cli.common import (
    TYPER_ENCORD_DATABASE_DIR,
    TYPER_SELECT_PROJECT_NAME,
    select_project_hash_from_name,
)
from encord_active.cli.config import app_config
from encord_active.cli.utils.prints import success_with_vizualise_command

import_cli = typer.Typer(rich_markup_mode="markdown")


@import_cli.command(name="predictions")
def import_predictions(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    project_name: Optional[str] = TYPER_SELECT_PROJECT_NAME,
    predictions_path: Path = typer.Argument(..., help="Path to a predictions file.", dir_okay=False, exists=True),
    coco: bool = typer.Option(False, help="Import a COCO results format file."),
    prediction_name: str = typer.Option("Prediction", help="Name of the prediction"),
):
    """
    [green bold]Imports[/green bold] a predictions file. The predictions should be using the `Prediction` model and be stored in a pkl file.

    If the `--coco` option is specified then the file should be a json following the COCO results format. :brain:
    """
    project_uuid = select_project_hash_from_name(database_dir, project_name or "")
    if coco:
        from encord_active.imports.op import import_coco_prediction

        import_coco_prediction(
            database_dir=database_dir,
            predictions_file_path=predictions_path,
            project_hash=project_uuid,
            prediction_name=prediction_name,
        )
    else:
        with open(predictions_path, "rb") as f:
            predictions = pickle.load(f)
        raise ValueError("Not supported yet!!!")


ENCORD_RICH_PANEL = "Encord Project Arguments"
COCO_RICH_PANEL = "COCO Project Arguments"
REQUIRED_WITH_COCO = f"[dark_red]{escape('[required with --coco]')}[/dark_red]"


@import_cli.command(name="project")
def import_project(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    # Encord
    encord_project_hash: Optional[str] = typer.Option(
        None,
        "--project-hash",
        help="Encord project hash of the project you wish to import. Leaving it blank will allow you to choose one interactively.",
        rich_help_panel=ENCORD_RICH_PANEL,
    ),
    # COCO
    coco: bool = typer.Option(False, help="Import a project from the COCO format.", rich_help_panel=COCO_RICH_PANEL),
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
    store_data_locally: bool = typer.Option(
        False,
        help="Store project data locally to avoid the need for on-demand download when visualizing and analyzing it.",
        rich_help_panel=ENCORD_RICH_PANEL,
    ),
):
    """
    [bold]Imports[/bold] a new project from Encord or a local COCO project. 📦

    Confirm with each help panel what information you will have to provide.
    """
    project_hash = uuid.uuid4()
    if encord_project_hash is not None:
        project_hash = uuid.UUID(encord_project_hash)

    if coco:
        from encord_active.imports.op import import_coco_project

        if not annotations:
            raise typer.BadParameter("`annotations` argument is missing")
        elif not images:
            raise typer.BadParameter("`images` argument is missing")

        import_coco_project(
            database_dir=database_dir,
            annotations_file_path=annotations,
            images_dir_path=images,
            store_symlinks=symlinks,
            store_data_locally=store_data_locally,
        )
    else:
        from encord_active.imports.op import import_encord_project

        ssh_key_path = app_config.get_or_query_ssh_key()
        import_encord_project(
            database_dir=database_dir,
            encord_project_hash=project_hash,
            ssh_key_path=ssh_key_path,
            store_data_locally=store_data_locally,
        )

    success_with_vizualise_command(database_dir, "The data is downloaded and the metrics are complete.")
