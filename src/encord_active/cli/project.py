import shutil
import uuid
from pathlib import Path
from typing import Optional

import rich
import typer
from encord import EncordUserClient
from encord.http.constants import RequestsSettings
from rich.panel import Panel
from tqdm import tqdm

from encord_active.cli.app_config import app_config
from encord_active.cli.common import (
    TYPER_ENCORD_DATABASE_DIR,
    TYPER_SELECT_PROJECT_NAME,
    select_project_hash_from_name,
)

project_cli = typer.Typer(rich_markup_mode="markdown")


@project_cli.command(name="download-data", short_help="Download all data locally for improved responsiveness.")
def download_data(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    project_name: Optional[str] = TYPER_SELECT_PROJECT_NAME,
) -> None:
    """
    Store project data locally to avoid the need for on-demand download when visualizing and analyzing it.
    """
    from sqlalchemy.sql.operators import is_, like_op
    from sqlmodel import Session, select

    from encord_active.db.models import Project, ProjectDataUnitMetadata, get_engine
    from encord_active.imports.local_files import download_to_local_file, get_local_file

    #
    project_hash = select_project_hash_from_name(database_dir, project_name or "")
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError(f"Project with hash: {project_hash} does not exist")
        encord_project = None
        if project.remote:
            encord_client = EncordUserClient.create_with_ssh_private_key(
                app_config.get_or_query_ssh_key(),
                requests_settings=RequestsSettings(max_retries=5),
            )
            encord_project = encord_client.get_project(str(project_hash))
        to_download = sess.exec(
            select(ProjectDataUnitMetadata).where(
                ProjectDataUnitMetadata.project_hash == project_hash,
                is_(ProjectDataUnitMetadata.data_uri, None)
                | like_op(ProjectDataUnitMetadata.data_uri, "http://%")
                | like_op(ProjectDataUnitMetadata.data_uri, "https://%"),
            )
        ).fetchall()
        for du in tqdm(to_download, desc="Downloading data to local disc"):
            if du.data_uri is not None:
                url: str = du.data_uri
            elif encord_project is not None:
                video, images = encord_project.get_data(str(du.data_hash), get_signed_url=True)
                if not images and video and video.get("file_link"):
                    url = str(video["file_link"])
                    du.data_uri_is_video = True
                else:
                    url_found = None
                    for image in images or []:
                        du_hash = uuid.UUID(image["image_hash"])
                        if du_hash == du.du_hash:
                            url_found = str(image["file_link"])
                            du.data_uri_is_video = False
                    if url_found is None:
                        raise ValueError(f"Encord data_hash does not have du_hash: {du.data_hash} => {du.du_hash}")
                    url = url_found
            else:
                raise ValueError(f"Data unit hash has not uri yet not an encord remote project: {du.du_hash}")
            du.data_uri = download_to_local_file(
                database_dir=database_dir,
                local_file=get_local_file(database_dir),
                url=url,
            )
            sess.add(du)
        sess.commit()


@project_cli.command(name="delete", short_help="Delete a project")
def delete_project(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    project_name: Optional[str] = TYPER_SELECT_PROJECT_NAME,
) -> None:
    """
    Delete a project from the encord active instance.
    """
    from encord_active.db.models import get_engine
    from encord_active.db.scripts.delete_project import delete_project_from_db

    #
    project_hash = select_project_hash_from_name(database_dir, project_name or "")
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    try:
        delete_project_from_db(
            engine=engine,
            project_hash=project_hash,
            error_on_missing=True,
        )
    except ValueError as e:
        rich.print(
            Panel(
                f"Could not delete project {project_hash} - not present in database\nError = {e}",
                title=":fire: No files found from data glob :fire:",
                expand=False,
                style="yellow",
            )
        )
        raise typer.Abort()
    else:
        rich.print(
            Panel(
                "Project deleted from database",
                expand=False,
                style="green",
            )
        )


@project_cli.command("serialize")
def serialize_project(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    project_name: Optional[str] = TYPER_SELECT_PROJECT_NAME,
    export_folder_name: str = typer.Option(
        None,
        "--export-folder-name",
        "-e",
        help="Export folder name",
    ),
) -> None:
    """
    Serialize the whole project state
    """
    from encord_active.db.models import get_engine
    from encord_active.exports.serialize import serialize_whole_project_state

    #
    project_hash = select_project_hash_from_name(database_dir, project_name or "")
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)

    export_folder = Path.cwd() / (export_folder_name if export_folder_name is not None else f"export-{project_hash}")
    serialize_whole_project_state(engine, database_dir, project_hash, export_folder)


@project_cli.command("deserialize")
def deserialize_project(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    export_folder: Path = typer.Option(
        ...,
        "--export-folder",
        "-e",
        help="Export folder name",
    ),
) -> None:
    """
    Populate a serialized project into the database.
    """
    from encord_active.db.models import get_engine
    from encord_active.imports.sandbox.deserialize import import_serialized_project

    #
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)

    store_folder = database_dir / "local-data" / export_folder.name
    db_json_file = store_folder / "db.json"
    if store_folder.exists():
        res = typer.confirm("Project folder already present in local-data, try import anyway?")
        if not res:
            return
        shutil.copytree(export_folder, store_folder, dirs_exist_ok=True)
    else:
        shutil.copytree(export_folder, store_folder)

    import_serialized_project(engine, db_json_file)
    db_json_file.unlink(missing_ok=False)
