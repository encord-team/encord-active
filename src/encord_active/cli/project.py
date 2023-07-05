from pathlib import Path
from typing import Dict

import typer
from tqdm.auto import tqdm

from encord_active.cli.utils.decorators import ensure_project
from encord_active.lib.common.data_utils import download_file, file_path_to_url
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.project import ProjectFileStructure

project_cli = typer.Typer(rich_markup_mode="markdown")


@project_cli.command(name="download-data", short_help="Download all data locally for improved responsiveness.")
@ensure_project()
def download_data(
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to the target project.", file_okay=False),
) -> None:
    """
    Store project data locally to avoid the need for on-demand download when visualizing and analyzing it.
    """
    project_file_structure = ProjectFileStructure(target)
    project_file_structure.local_data_store.mkdir(exist_ok=True)
    with PrismaConnection(project_file_structure) as conn:
        batch_updates: Dict[str, str] = {}
        for label in tqdm(project_file_structure.iter_labels(cache_db=conn), desc="Downloading project data locally"):
            data_units = label.get_label_row_json(cache_db=conn).get("data_units")
            for db in label.iter_data_unit():
                du = data_units.get(db.du_hash, {"data_title": ""})
                file_path = (project_file_structure.local_data_store / db.du_hash).with_suffix(
                    Path(du["data_title"]).suffix
                )
                data_uri = file_path_to_url(file_path, project_dir=project_file_structure.project_dir)
                download_file(
                    db.signed_url,
                    project_file_structure.project_dir,
                    file_path,
                    cache=False,  # Disable cache symlink tricks
                )
                batch_updates[db.du_hash] = data_uri
        with conn.batch_() as batcher:
            for du_hash, local_uri in batch_updates.items():
                batcher.dataunit.update_many(
                    where={
                        "data_hash": du_hash,
                    },
                    data={"data_uri": local_uri},
                )
