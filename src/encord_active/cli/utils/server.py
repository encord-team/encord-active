from pathlib import Path

import rich

from encord_active.cli.app_config import app_config
from encord_active.server.start_server import start


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

    # Launch server
    data_dir = target.expanduser().absolute()
    rich.print(f"[green] Server starting on [blue]http://localhost:{port}[/blue]")
    start(data_dir, port=port, reload=app_config.is_dev)
