import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import rich
import typer
from rich.markup import escape

from encord_active.cli.config import app_config
from encord_active.cli.utils.decorators import ensure_project

print_cli = typer.Typer(rich_markup_mode="markdown")

state: Dict[str, Any] = {}


@print_cli.command(name="encord-projects")
def print_encord_projects(
    query: Optional[str] = typer.Option(None, help="Optional fuzzy title filter; SQL syntax."),
):
    """
    Print the mapping between `project_hash`es of your Encord projects and their titles.

    You can query projects by title with the SQL fuzzy syntax. To look for a title including "validation" you would do:

    > encord-active print encord-projects --query "%validation%"

    """
    from encord_active.lib.encord.utils import get_projects_json

    json_projects = get_projects_json(app_config.get_or_query_ssh_key(), query)
    if state.get("json_output"):
        Path("./encord-projects.json").write_text(json_projects, encoding="utf-8")
    else:
        rich.print(escape(json_projects))


@print_cli.command(name="ontology")
@ensure_project
def print_ontology(
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to a local project.", file_okay=False),
):
    """
    [bold]Prints[/bold] an ontology mapping between the class name to the `featureNodeHash` JSON format.
    """
    from rich.panel import Panel

    from encord_active.lib.project.project_file_structure import ProjectFileStructure

    fs = ProjectFileStructure(target)
    if not fs.ontology.is_file():
        rich.print(
            Panel(
                """
Couldn't identify a project ontology. The reason for this may be that you have a very old project. Please try re-importing the project.
                """,
                title=":exclamation: :exclamation: ",
                style="yellow",
                expand=False,
            )
        )

        raise typer.Exit()

    objects = json.loads(fs.ontology.read_text())["objects"]

    ontology = {o["name"].lower(): o["featureNodeHash"] for o in objects}
    json_ontology = json.dumps(ontology, indent=2)

    if state.get("json_output"):
        Path("./ontology.json").write_text(json_ontology, encoding="utf-8")
        rich.print("Stored mapping in [blue]`./ontology.json`")
    else:
        rich.print(escape(json_ontology))


@print_cli.command(name="data-mapping")
@ensure_project
def print_data_mapping(
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to a local project.", file_okay=False),
    limit: int = typer.Option(None, help="Limit the result to the first `limit` data hashes"),
):
    """
    [bold]Prints[/bold] a mapping between `data_hashes` and their corresponding `filename`
    """
    mapping: Dict[str, str] = {}

    for label in (target / "data").iterdir():
        if not label.is_dir() and not (label / "label_row.json").is_file():
            continue

        label_row = json.loads((label / "label_row.json").read_text())
        mapping = {
            **mapping,
            **{data_hash: value["data_title"] for data_hash, value in label_row["data_units"].items()},
        }
        if limit and len(mapping) > limit:
            break

    if limit and limit < len(mapping):
        mapping = {k: v for i, (k, v) in enumerate(mapping.items()) if i < limit}

    json_mapping = json.dumps(mapping, indent=2)

    if state.get("json_output"):
        Path("./data_mapping.json").write_text(json_mapping, encoding="utf-8")
        rich.print("Stored mapping in [blue]`./data_mapping.json`")
    else:
        rich.print(escape(json_mapping))


@print_cli.command(name="system-info")
def print_system_info():
    """
    [bold]Prints[/bold] the information of the system for the purpose of bug reporting.
    """
    import platform

    import psutil

    def get_size(bytes, suffix="B"):
        """
        Scale bytes to its proper format
        e.g:
            1253656 => '1.20MB'
            1253656678 => '1.17GB'
        """
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor

    print("System Information:")
    uname = platform.uname()
    print(f"\tSystem: {uname.system}")
    print(f"\tRelease: {uname.release}")
    print(f"\tMachine: {uname.machine}")
    print(f"\tProcessor: {uname.processor}")
    print(f"\tPython: {sys.version}")
    print("\nCPU Info:")
    print("\tPhysical cores:", psutil.cpu_count(logical=False))
    print("\tTotal cores:", psutil.cpu_count(logical=True))
    print(f"\tTotal CPU Usage: {psutil.cpu_percent()}%")
    print("\nMemory Information:")
    svmem = psutil.virtual_memory()
    print(f"\tTotal: {get_size(svmem.total)}")
    print(f"\tAvailable: {get_size(svmem.available)}")
    print(f"\tUsed: {get_size(svmem.used)}")


@print_cli.callback()
def main(json: bool = False):  # pylint: disable=redefined-outer-name
    """
    [green bold]Print[/green bold] useful information 🖨️
    """
    state["json_output"] = json
