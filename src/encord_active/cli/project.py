import uuid
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from encord_active.cli.common import (
    TYPER_ENCORD_DATABASE_DIR,
    select_project_hash_from_name,
)
from encord_active.lib.encord.utils import get_encord_project

project_cli = typer.Typer(rich_markup_mode="markdown")


@project_cli.command(name="download-data", short_help="Download all data locally for improved responsiveness.")
def download_data(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    project_name: Optional[str] = typer.Option(None, help="Name of the chosen project."),
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
        if project.project_remote_ssh_key_path is not None:
            encord_project = get_encord_project(
                ssh_key_path=project.project_remote_ssh_key_path, project_hash=str(project_hash)
            )
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
                if video is not None:
                    url = str(video["file_link"])
                    du.data_uri_is_video = True
                else:
                    url_found = None
                    for image in images or []:
                        du_hash = uuid.UUID(image["data_hash"])
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
