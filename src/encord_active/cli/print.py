import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

import rich
import typer
from rich.markup import escape

from encord_active.cli.common import (
    TYPER_ENCORD_DATABASE_DIR,
    TYPER_SELECT_PROJECT_NAME,
    select_project_hash_from_name,
)
from encord_active.cli.config import app_config
from encord_active.lib.common.data_utils import url_to_file_path

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
def print_ontology(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    project_name: Optional[str] = TYPER_SELECT_PROJECT_NAME,
):
    """
    [bold]Prints[/bold] an ontology mapping between the class name to the `featureNodeHash` JSON format.
    """
    from encord.ontology import OntologyStructure
    from sqlmodel import Session, select

    from encord_active.db.models import Project, get_engine
    from encord_active.lib.model_predictions.writer import (
        iterate_classification_attribute_options,
    )

    #
    project_hash = select_project_hash_from_name(database_dir, project_name or "")
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError(f"Project hash does not exist: {project_hash}")
    project_ontology = project.ontology

    ontology_structure = OntologyStructure.from_dict(project_ontology)

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
def print_data_mapping(
    database_dir: Path = TYPER_ENCORD_DATABASE_DIR,
    project_name: Optional[str] = TYPER_SELECT_PROJECT_NAME,
    limit: Optional[int] = typer.Option(None, help="Limit the result to the first `limit` data hashes"),
):
    """
    [bold]Prints[/bold] a mapping between `data_hashes` and their corresponding `filename`.
    """
    from sqlalchemy.sql.operators import is_not
    from sqlmodel import Session, select

    from encord_active.db.models import Project, ProjectDataUnitMetadata, get_engine

    #
    project_hash = select_project_hash_from_name(database_dir, project_name or "")
    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError(f"Project with hash: {project_hash} does not exist")
        local_files = sess.exec(
            select(ProjectDataUnitMetadata)
            .where(ProjectDataUnitMetadata.project_hash == project_hash, is_not(ProjectDataUnitMetadata.data_uri, None))
            .order_by(ProjectDataUnitMetadata.du_hash, ProjectDataUnitMetadata.frame)
            .limit(limit)
        ).fetchall()
    mapping: Dict[str, str] = {}
    local_files_seen: Set[str] = set()

    for local_file in local_files:
        if limit is not None and len(mapping) >= limit:
            break
        if local_file.data_uri is None:
            continue
        if local_file.data_uri in local_files_seen:
            continue
        local_files_seen.add(local_file.data_uri)
        map_url_str = local_file.data_uri
        opt_path = url_to_file_path(local_file.data_uri, database_dir)
        if opt_path is not None:
            if opt_path.is_relative_to(database_dir):
                map_url_str = opt_path.relative_to(database_dir).as_posix()
            else:
                map_url_str = opt_path.as_posix()
        mapping[str(local_file.du_hash)] = map_url_str

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
