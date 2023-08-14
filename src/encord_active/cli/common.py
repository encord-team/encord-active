import uuid
from pathlib import Path

import rich
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


def select_project_hash_from_name(database_dir: Path, project_name: str) -> uuid.UUID:
    try:
        # Check if this is actually a uuid.
        return uuid.UUID(project_name)
    except ValueError:
        pass

    from sqlmodel import Session, select

    from encord_active.db.models import Project, get_engine

    #
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        # Exact search
        unique_project = sess.exec(select(Project.project_hash).where(Project.project_name == project_name)).fetchall()
        if len(unique_project) == 1:
            return unique_project[0]
        elif len(unique_project) > 1:
            from InquirerPy import inquirer as i

            project_uuid_str = i.select(
                message="Choose a project uuid, multiple projects with that name exist",
                choices=[str(x) for x in unique_project],
                vi_mode=True,
            ).execute()
            if project_uuid_str is None:
                rich.print("No project was selected.")
                raise typer.Abort()
            else:
                return uuid.UUID(project_uuid_str)

        # Fuzzy search
        fuzzy_project = sess.exec(
            select(Project.project_name, Project.project_hash).where(
                Project.project_name.like("%" + project_name + "%")  # type: ignore
            )
        ).fetchall()
        if len(fuzzy_project) == 1:
            return fuzzy_project[0][1]
        elif len(fuzzy_project) > 1:
            from InquirerPy import inquirer as i

            project_name_uuid_str = i.select(
                message="Choose a project uuid, multiple projects with that name exist",
                choices=[f"{name}: {p_uuid}" for name, p_uuid in fuzzy_project],
                vi_mode=True,
            ).execute()
            if project_name_uuid_str is None:
                rich.print("No project was selected.")
                raise typer.Abort()
            else:
                project_uuid_str = project_name_uuid_str.split(":")[-1].strip()
                return uuid.UUID(project_uuid_str)

    rich.print("Failed to select a project.")
    raise typer.Abort()
