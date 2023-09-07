from pathlib import Path
from typing import Optional

import rich

from encord_active.cli.app_config import app_config
from encord_active.cli.utils.decorators import find_child_projects, is_project
from encord_active.lib.db.data.run_data_migrations import run_data_migrations
from encord_active.lib.db.prisma_init import (
    did_schema_change,
    ensure_prisma_db,
    generate_prisma_client,
)
from encord_active.lib.project.project_file_structure import ProjectFileStructure
from encord_active.server.start_server import start


def ensure_safe_project(root_path: Path, final_data_version: Optional[int] = None):
    paths = [root_path] if is_project(root_path) else find_child_projects(root_path)

    for path in paths:
        rich.print(f"[yellow]Migrating project at path: {path}...")
        project_file_structure = ProjectFileStructure(path)

        from encord_active.lib.db.merged_metrics import (
            ensure_initialised_merged_metrics,
        )

        ensure_initialised_merged_metrics(project_file_structure)
        ensure_prisma_db(project_file_structure.prisma_db)
        run_data_migrations(project_file_structure, final_data_version=final_data_version)


def is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def launch_server_app(target: Path, port: int):
    if is_port_in_use(port):
        import typer

        rich.print(
            f"[orange1]Port [blue]{port}[/blue] already in use. Try changing the `[blue]--port[/blue]` option.[/orange1]"
        )
        raise typer.Exit()
    else:
        rich.print("[yellow]Bear with us, this might take a short while...")

    if did_schema_change():
        generate_prisma_client()
    ensure_safe_project(target)
    data_dir = target.expanduser().absolute()
    rich.print(f"[green] Server starting on [blue]http://localhost:{port}[/blue]")
    start(data_dir, port=port, reload=app_config.is_dev)
