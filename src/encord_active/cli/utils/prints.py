from pathlib import Path

import rich
from rich.markup import escape
from rich.panel import Panel


def success_with_visualise_command(
    project_path: Path, success_text: str = "The data is downloaded and the metrics are complete."
):
    cwd = Path.cwd()
    cd_dir = project_path.relative_to(cwd) if project_path.is_relative_to(cwd) else project_path

    panel = Panel(
        f"""
{success_text}

💡 Hint: You can open your project in Encord Active with the following commands

[cyan]cd "{escape(cd_dir.as_posix())}"
encord-active visualise[/cyan]

to open Encord Active and see your project.
    """,
        title="🌟 Success 🌟",
        style="green",
        expand=False,
    )
    rich.print(panel)
