import subprocess
import sys
from os import environ
from pathlib import Path

import rich
from streamlit.web import cli as stcli

from encord_active.app.app_config import app_config
from encord_active.cli.utils.decorators import find_child_projects, is_project
from encord_active.lib.db.data.run_data_migrations import run_data_migrations
from encord_active.lib.db.merged_metrics import ensure_initialised_merged_metrics
from encord_active.lib.db.prisma_init import (
    did_schema_change,
    ensure_prisma_db,
    generate_prisma_client,
)
from encord_active.lib.project.project_file_structure import ProjectFileStructure
from encord_active.lib.versioning.git import GitVersioner


def ensure_safe_project(root_path: Path):
    paths = [root_path] if is_project(root_path) else find_child_projects(root_path)

    for path in paths:
        rich.print(f"[yellow]Migrating project at path: {path}...")
        versioner = GitVersioner(path)
        if versioner.available:
            versioner.jump_to("latest")
        project_file_structure = ProjectFileStructure(path)
        ensure_initialised_merged_metrics(project_file_structure)
        ensure_prisma_db(project_file_structure.prisma_db)
        run_data_migrations(project_file_structure)


def is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def launch_streamlit_app(target: Path):
    if is_port_in_use(8501):
        import typer

        rich.print("[red]Port already in use...")
        raise typer.Exit()
    else:
        rich.print("[yellow]Bear with us, this might take a short while...")

    if did_schema_change():
        generate_prisma_client()
    ensure_safe_project(target)
    streamlit_page = (Path(__file__).parents[2] / "app" / "streamlit_entrypoint.py").expanduser().absolute()
    data_dir = target.expanduser().absolute().as_posix()
    sys.argv = ["streamlit", "run", streamlit_page.as_posix(), data_dir]

    # NOTE: we need to set PYTHONPATH for file watching
    python_path = Path(__file__).parents[2]
    environ["PYTHONPATH"] = (python_path).as_posix()

    server_args = [(python_path / "server" / "start_server.py").as_posix(), target.as_posix()]

    if app_config.is_dev:
        sys.argv.insert(3, "--server.fileWatcherType")
        sys.argv.insert(4, "auto")
        server_args.append("--reload")

    subprocess.Popen([sys.executable, *server_args])

    stcli.main()  # pylint: disable=no-value-for-parameter
