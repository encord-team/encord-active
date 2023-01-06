import pickle
from pathlib import Path
from typing import Optional

import typer

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


@import_cli.command(name="project")
def import_project(
    images_dir: Optional[Path] = typer.Argument(
        None,
        help="Path to the directory containing the dataset images. (only required with `--cooc`)",
        file_okay=False,
        exists=True,
    ),
    annotations_file: Optional[Path] = typer.Argument(
        None,
        help="Path to the file containing the dataset annotations. (only required with `--cooc`)",
        dir_okay=False,
        exists=True,
    ),
    encord_project_hash: Optional[str] = typer.Option(
        None, "--project_hash", help="Encord project hash of the project you wish to import."
    ),
    target: Path = typer.Option(
        Path.cwd(), "--target", "-t", help="Directory where the project would be saved.", file_okay=False
    ),
    coco: bool = typer.Option(False, help="Import a project from the coco format"),
):
    """
    [bold]Imports[/bold] a new project from Encord or a local coco project 📦
    """
    if coco:
        from encord_active.cli.utils.coco import import_coco_project

        if not annotations_file:
            raise typer.BadParameter("`annotations_file` argument is missing")
        elif not images_dir:
            raise typer.BadParameter("`images_dir` argument is missing")

        ssh_key_path = app_config.get_or_query_ssh_key()
        import_coco_project(images_dir, annotations_file, target, ssh_key_path)
    else:
        from encord_active.cli.utils.encord import import_encord_project

        ssh_key_path = app_config.get_or_query_ssh_key()
        import_encord_project(ssh_key_path, target, encord_project_hash)
