from pathlib import Path
from typing import Optional

import encord.exceptions
import rich
import typer
import yaml
from encord.objects.ontology_structure import OntologyStructure
from rich.markup import escape
from rich.panel import Panel

from encord_active.lib.common.data_utils import collect_async
from encord_active.lib.db.helpers.tags import populate_tags_with_nested_classifications
from encord_active.lib.encord.utils import get_client, get_encord_projects
from encord_active.lib.labels.ontology import get_nested_radio_and_checklist_hashes
from encord_active.lib.metrics.execute import run_metrics
from encord_active.lib.metrics.io import fill_metrics_meta_with_builtin_metrics
from encord_active.lib.metrics.metadata import update_metrics_meta
from encord_active.lib.project.project_file_structure import ProjectFileStructure

PROJECT_HASH_REGEX = r"([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})"


def suggest_tagging_data(ontology: OntologyStructure) -> set[str]:
    """
    Looks for objects with nested classifications.
    If there are any, will prompt user if classifications should be used as tags.
    Args:
        ontology: the project ontology which will be traversed.

    Returns: The feature node hashes of the ontology objects that have nested
    classifications or an empty set.
    """
    option_answer_hashes = get_nested_radio_and_checklist_hashes(ontology)
    if option_answer_hashes and typer.confirm(
        "Your project has nested classifications. Would you like to include them as tags?"
    ):
        return option_answer_hashes
    return set()


def import_encord_project(
    ssh_key_path: Path,
    target: Path,
    encord_project_hash: Optional[str],
    store_data_locally: bool,
) -> Path:
    from .server import ensure_safe_project

    client = get_client(ssh_key_path)

    if encord_project_hash:
        project_hash = encord_project_hash
    else:
        from InquirerPy import inquirer as i
        from InquirerPy.base.control import Choice

        projects = get_encord_projects(ssh_key_path)
        if not projects:
            rich.print(
                Panel(
                    """
Couldn't find any projects to import.
Check that you have the correct ssh key set up and available projects on [blue]https://app.encord.com/projects[/blue].
                    """,
                    title="⚡️ [red]Failed to find projects[/red] ⚡️",
                    style="yellow",
                    expand=False,
                )
            )
            exit()

        choices = list(map(lambda p: Choice(p.project_hash, name=p.title), projects))
        project_hash = i.fuzzy(
            message="What project would you like to import?",
            choices=choices,
            vi_mode=True,
            multiselect=False,
            instruction="💡 Type a (fuzzy) search query to find the project you want to import.",
        ).execute()

    try:
        project = client.get_project(project_hash)
        _ = project.title
    except encord.exceptions.AuthorisationError:
        rich.print("⚡️ [red]You don't have access to the project, sorry[/red] 😫")
        exit()

    project_path = target / project.title.lower().replace(" ", "-")
    project_path.mkdir(exist_ok=True, parents=True)
    project_file_structure = ProjectFileStructure(project_path)

    ontology = OntologyStructure.from_dict(project.ontology)
    option_hashes_to_tag = suggest_tagging_data(ontology)

    meta_data = {
        "project_title": project.title,
        "project_description": project.description,
        "project_hash": project.project_hash,
        "ssh_key_path": ssh_key_path.as_posix(),
        "has_remote": True,
        "nested_attributes_as_tags": bool(option_hashes_to_tag),
        "store_data_locally": store_data_locally,
    }
    yaml_str = yaml.dump(meta_data)
    project_file_structure.project_meta.write_text(yaml_str, encoding="utf-8")

    # attach builtin metrics to the project
    metrics_meta = fill_metrics_meta_with_builtin_metrics()
    update_metrics_meta(project_file_structure, metrics_meta)

    rich.print("Stored the following data:")
    rich.print(f"[magenta]{escape(yaml_str)}")
    rich.print(f'In file: [blue]"{escape(project_file_structure.project_meta.as_posix())}"')

    has_uninitialized_rows = not all(row["label_hash"] is not None for row in project.label_rows)
    if has_uninitialized_rows and typer.confirm(
        """Would you like to include uninitialized label rows?
NOTE: this will affect the results of 'encord.Project.list_label_rows()' as every label row will now have a label_hash.
        """
    ):
        untoched_data = list(filter(lambda x: x.label_hash is None, project.list_label_rows_v2()))
        collect_async(lambda x: x.initialise_labels(), untoched_data, desc="Preparing uninitialized label rows")
        project.refetch_data()
        rich.print()

    rich.print("Now downloading data and running metrics")

    run_metrics(data_dir=project_path)
    ensure_safe_project(project_file_structure.project_dir)

    if option_hashes_to_tag:
        populate_tags_with_nested_classifications(project_file_structure, option_hashes_to_tag)

    return project_path
