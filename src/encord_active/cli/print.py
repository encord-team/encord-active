import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import rich
import typer
from rich.markup import escape

from encord_active.cli.config import app_config
from encord_active.cli.utils.decorators import ensure_project
from encord_active.lib.project import ProjectFileStructure

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
    from encord_active.lib.encord.utils import ProjectQuery, get_projects_json

    project_query = None if query is None else ProjectQuery(title_like=query)
    json_projects = get_projects_json(app_config.get_or_query_ssh_key(), project_query)
    if state.get("json_output"):
        output_file_path = Path("./encord-projects.json").resolve()
        output_file_path.write_text(json_projects, encoding="utf-8")
        rich.print(f"Mapping stored in [blue]`{output_file_path.as_posix()}`")
    else:
        rich.print(escape(json_projects))


@print_cli.command(name="ontology")
@ensure_project()
def print_ontology(
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to a local project.", file_okay=False),
):
    """
    [bold]Prints[/bold] an ontology mapping between the class name to the `featureNodeHash` JSON format.
    """
    from encord.ontology import OntologyStructure
    from rich.panel import Panel

    from encord_active.lib.model_predictions.writer import (
        iterate_classification_attribute_options,
    )

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

    ontology_structure = OntologyStructure.from_dict(json.loads(fs.ontology.read_text()))

    classifications = {
        option.value: hashes.dict()
        for (hashes, (_, _, option)) in iterate_classification_attribute_options(ontology_structure)
    }

    objects = {o.name.lower(): o.feature_node_hash for o in ontology_structure.objects}

    json_ontology = json.dumps({"objects": objects, "classifications": classifications}, indent=2)

    if state.get("json_output"):
        output_file_path = Path("./ontology_output.json").resolve()
        output_file_path.write_text(json_ontology, encoding="utf-8")
        rich.print(f"Mapping stored in [blue]`{output_file_path.as_posix()}`")
    else:
        rich.print(escape(json_ontology))


@print_cli.command(name="data-mapping")
@ensure_project()
def print_data_mapping(
    target: Path = typer.Option(Path.cwd(), "--target", "-t", help="Path to a local project.", file_okay=False),
    limit: int = typer.Option(None, help="Limit the result to the first `limit` data hashes"),
):
    """
    [bold]Prints[/bold] a mapping between `data_hashes` and their corresponding `filename`.
    """
    mapping: Dict[str, str] = {}
    project_file_structure = ProjectFileStructure(target)

    for label_row_structure in project_file_structure.iter_labels():
        label_row = label_row_structure.label_row_json
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
        output_file_path = Path("./data_mapping.json").resolve()
        output_file_path.write_text(json_mapping, encoding="utf-8")
        rich.print(f"Mapping stored in [blue]`{output_file_path.as_posix()}`")
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
def main(json: bool = typer.Option(False, help="Save output to a json file.")):  # pylint: disable=redefined-outer-name
    """
    [green bold]Print[/green bold] useful information 🖨️
    """
    state["json_output"] = json
