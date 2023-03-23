import json
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Tuple

import streamlit as st
from encord.ontology import OntologyStructure
from encord_active_components.components.projects_page import (
    OutputAction,
    Project,
    ProjectStats,
    projects_page,
)
from PIL import Image
from streamlit.elements.image import AtomicImage, image_to_url

from encord_active.app.actions_page.export_filter import render_progress_bar
from encord_active.app.common.state_hooks import UseState
from encord_active.cli.utils.decorators import (
    find_child_projects,
    is_project,
    try_find_parent_project,
)
from encord_active.lib.common.image_utils import show_image_and_draw_polygons
from encord_active.lib.metrics.metric import AnnotationType
from encord_active.lib.metrics.utils import load_metric_metadata
from encord_active.lib.model_predictions.writer import (
    iterate_classification_attribute_options,
)
from encord_active.lib.project.metadata import fetch_project_meta
from encord_active.lib.project.project_file_structure import ProjectFileStructure
from encord_active.lib.project.sandbox_projects import (
    PREBUILT_PROJECTS,
    fetch_prebuilt_project,
    unpack_archive,
)
from encord_active.lib.versioning.git import GitVersioner


def project_list(path: Path):
    child_projects = find_child_projects(path)
    if is_project(path):
        return [path, *child_projects]
    else:
        parent_project = try_find_parent_project(path)
        return [parent_project] if parent_project else child_projects


def prevent_detached_versions(path: Path):
    paths = [path] if is_project(path) else find_child_projects(path)
    for path in paths:
        GitVersioner(path).jump_to("latest")


def image_url(image: AtomicImage, project_hash: str):
    id = f"project-{project_hash}-image"
    return image_to_url(image, -1, False, "RGB", "JPEG", id)


# TODO: repalce me with something smarter than just the first image
@st.cache_data(show_spinner=False)
def get_first_image_with_polygons(project_path: Path):
    project_structure = ProjectFileStructure(project_path)
    label_structure = next(project_structure.iter_labels())
    label_hash = label_structure.path.stem
    image_path = next(label_structure.iter_data_unit()).path
    data_hash = image_path.stem
    id = f"{label_hash}_{data_hash}_00000"

    return show_image_and_draw_polygons(id, project_structure)


# TODO: there must be a better way to get these numbers. would be much easier
# if we store everything in the DB.
@st.cache_data(show_spinner=False)
def get_project_stats(project_path: Path):
    project_structure = ProjectFileStructure(project_path)
    label_count = 0
    data_count = 0

    for meta_path in project_structure.metrics.glob("*.meta.json"):
        meta = load_metric_metadata(meta_path)
        if meta.annotation_type == AnnotationType.NONE:
            data_count = max(data_count, meta.stats.num_rows)
        else:
            label_count = max(label_count, meta.stats.num_rows)

    ontology_structure = OntologyStructure.from_dict(json.loads(project_structure.ontology.read_text()))
    objects_count = len(ontology_structure.objects)
    classifications_count = len(list(iterate_classification_attribute_options(ontology_structure)))

    return ProjectStats(dataUnits=data_count, labels=label_count, classes=(objects_count + classifications_count))


class GetProjectsResult(NamedTuple):
    local: dict[str, Project]
    sandbox: dict[str, Project]
    local_paths: dict[str, Path]


def get_projects(path: Path):
    prevent_detached_versions(path)
    project_metas = {project: fetch_project_meta(project) for project in project_list(path)}

    local_projects = {
        project["project_hash"]: Project(
            name=project.get("project_title") or project_path.name,
            hash=project["project_hash"],
            downloaded=True,
            imageUrl=image_url(get_first_image_with_polygons(project_path), project["project_hash"]),
            stats=get_project_stats(project_path),
        )
        for project_path, project in project_metas.items()
    }

    sandbox_projects = {
        data["hash"]: Project(
            name=name,
            hash=data["hash"],
            downloaded=data["hash"] in local_projects,
            stats=data["stats"],
            imageUrl=image_url(Image.open(data["image_path"].resolve().as_posix()), data["hash"]),
        )
        for name, data in PREBUILT_PROJECTS.items()
    }

    local_project_paths = {project["project_hash"]: path for path, project in project_metas.items()}

    return GetProjectsResult(local_projects, sandbox_projects, local_project_paths)


def handle_download_sandbox_project(project_name: str, path: Path):
    project_dir = path / project_name
    project_dir.mkdir(exist_ok=True)

    render_progress_bar()
    label = st.empty()
    progress, clear = render_progress_bar()
    label.text(f"Downloading {project_name}")

    archive_path = fetch_prebuilt_project(project_name, project_dir, unpack=False, progress_callback=progress)
    with st.spinner():
        clear()
        label.text(f"Unpacking {project_name}")
        unpack_archive(archive_path, project_dir)

    label.empty()


def render_projects_page(
    select_project: Callable[[str, bool], None],
    projects: GetProjectsResult,
    download_path: Path,
):
    user_projects = [project for hash, project in projects.local.items() if hash not in projects.sandbox]
    output_state = UseState[Optional[Tuple[OutputAction, str]]](None, "PROJECTS_PAGE_OUTPUT")
    output = projects_page(user_projects=user_projects, sandbox_projects=list(projects.sandbox.values()))

    if output and output != output_state.value:
        output_state.set(output)
        action, payload = output

        if payload and action in [
            OutputAction.SELECT_USER_PROJECT,
            OutputAction.SELECT_SANDBOX_PROJECT,
        ]:
            refetch_projects = False
            if payload not in projects.local:
                handle_download_sandbox_project(projects.sandbox[payload]["name"], download_path)
                refetch_projects = True

            select_project(payload, refetch_projects)
        else:
            st.error("Something went wrong")