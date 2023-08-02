from pathlib import Path

import typer

TYPER_ENCORD_DATABASE_DIR = typer.Option(
    Path.cwd(),
    "--target",  # "--database-dir",
    "-t",  # "-d",
    help="Path to the encord database directory",
    file_okay=False,
)

TYPER_ENCORD_PROJECT_HASH = typer.Argument(
    ...,
)
