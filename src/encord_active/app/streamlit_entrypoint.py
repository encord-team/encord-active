import argparse
import shutil
from functools import reduce
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Tuple

import streamlit as st
from encord_active_components.components.pages_menu import MenuItem
from encord_active_components.components.pages_menu import (
    OutputAction as PagesMenuOutputAction,
)
from encord_active_components.components.pages_menu import Project, pages_menu
from encord_active_components.components.projects_page import (
    OutputAction as ProjectsPageOutputAction,
)
from encord_active_components.components.projects_page import (
    ProjectStats,
    projects_page,
)
from PIL import Image
from streamlit.elements.image import image_to_url

from encord_active.app.actions_page.export_balance import export_balance
from encord_active.app.actions_page.export_filter import (
    export_filter,
    render_progress_bar,
)
from encord_active.app.actions_page.versioning import is_latest, version_form
from encord_active.app.common.components.help.help import render_help
from encord_active.app.common.state import State, get_state, has_state, refresh
from encord_active.app.common.state_hooks import UseState, use_memo
from encord_active.app.common.utils import set_page_config
from encord_active.app.model_quality.sub_pages.false_negatives import FalseNegativesPage
from encord_active.app.model_quality.sub_pages.false_positives import FalsePositivesPage
from encord_active.app.model_quality.sub_pages.metrics import MetricsPage
from encord_active.app.model_quality.sub_pages.performance_by_metric import (
    PerformanceMetric,
)
from encord_active.app.model_quality.sub_pages.true_positives import TruePositivesPage
from encord_active.app.views.metrics import explorer, summary
from encord_active.app.views.model_quality import model_quality
from encord_active.cli.utils.decorators import (
    find_child_projects,
    is_project,
    try_find_parent_project,
)
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.metrics.utils import MetricScope
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.project.metadata import fetch_project_meta
from encord_active.lib.project.sandbox_projects import (
    PREBUILT_PROJECTS,
    fetch_prebuilt_project,
    unpack_archive,
)
from encord_active.lib.versioning.git import GitVersioner

pages = {
    "Data Quality": {"Summary": summary(MetricScope.DATA_QUALITY), "Explorer": explorer(MetricScope.DATA_QUALITY)},
    "Label Quality": {"Summary": summary(MetricScope.LABEL_QUALITY), "Explorer": explorer(MetricScope.LABEL_QUALITY)},
    "Model Quality": {
        "Metrics": model_quality(MetricsPage()),
        "Performance By Metric": model_quality(PerformanceMetric()),
        "True Positives": model_quality(TruePositivesPage()),
        "False Positives": model_quality(FalsePositivesPage()),
        "False Negatives": model_quality(FalseNegativesPage()),
    },
    "Actions": {"Filter & Export": export_filter, "Balance & Export": export_balance, "Versioning": version_form},
}

DEFAULT_PAGE_PATH = ["Data Quality", "Summary"]
SEPARATOR = "#"


def provide_backcompatibility_for_old_predictions():
    """
    TODO: For back compatibility of the predictions/ folder. Can be removed later (3+ months).
    If there is no object folder, and prediction files/folders are inside predictions/ folder
    Create a predictions/object folder and move all prediction-related files inside
    """

    prediction_items = []
    object_folder_not_exist = True
    prediction_file_exist = False
    if not get_state().project_paths.predictions.is_dir():
        return
    for item in get_state().project_paths.predictions.iterdir():
        if item.name == MainPredictionType.OBJECT.value:
            object_folder_not_exist = False
            break
        else:
            if item.name == "predictions.csv":
                prediction_file_exist = True
            if item.name != MainPredictionType.CLASSIFICATION.value:
                prediction_items.append(item)

    if object_folder_not_exist and prediction_file_exist:
        object_dir = get_state().project_paths.predictions / MainPredictionType.OBJECT.value
        object_dir.mkdir()
        for item in prediction_items:
            if item.name != ".DS_Store":
                shutil.move(item.as_posix(), object_dir.as_posix())


def to_item(k, v, parent_key: Optional[str] = None):
    # NOTE: keys must be unique for the menu to render properly
    composite_key = SEPARATOR.join(filter(None, [parent_key, k]))
    item = MenuItem(key=composite_key, label=k, children=None)
    if isinstance(v, dict):
        item["children"] = to_items(v, parent_key=composite_key)
    return item


def to_items(d: dict, parent_key: Optional[str] = None):
    return [to_item(k, v, parent_key) for k, v in d.items()]


def project_list(path: Path):
    if is_project(path):
        return [path]
    else:
        parent_project = try_find_parent_project(path)
        return [parent_project] if parent_project else find_child_projects(path)


def prevent_detached_versions(path: Path):
    paths = [path] if is_project(path) else find_child_projects(path)
    for path in paths:
        GitVersioner(path).jump_to("latest")


class GetProjectsResult(NamedTuple):
    local: dict[str, Project]
    sandbox: dict[str, Project]
    local_paths: dict[str, Path]


def image_url(path: Path, id: str):
    image = Image.open(path.resolve().as_posix())
    return image_to_url(image, -1, False, "RGB", "JPEG", id)


def get_projects(path: Path):
    prevent_detached_versions(path)
    project_metas = {project: fetch_project_meta(project) for project in project_list(path)}

    local_projects = {
        project["project_hash"]: Project(
            name=project.get("project_title") or path.name,
            hash=project["project_hash"],
            downloaded=True,
            imageUrl="",
            stats=ProjectStats(dataUnits=1000, labels=14566, classes=8),
        )
        for path, project in project_metas.items()
    }

    sandbox_projects = {
        data["hash"]: Project(
            name=name,
            hash=data["hash"],
            downloaded=data["hash"] in local_projects,
            stats=ProjectStats(dataUnits=1000, labels=14566, classes=8),
            imageUrl=image_url(data["image_path"], f"project-{data['hash']}-image"),
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
    output_state = UseState[Optional[Tuple[ProjectsPageOutputAction, str]]](None, "PROJECTS_PAGE_OUTPUT")
    output = projects_page(user_projects=user_projects, sandbox_projects=list(projects.sandbox.values()))

    if output and output != output_state.value:
        output_state.set(output)
        action, payload = output

        if payload and action in [
            ProjectsPageOutputAction.SELECT_USER_PROJECT,
            ProjectsPageOutputAction.SELECT_SANDBOX_PROJECT,
        ]:
            refetch_projects = False
            if payload not in projects.local:
                handle_download_sandbox_project(projects.sandbox[payload]["name"], download_path)
                refetch_projects = True

            select_project(payload, refetch_projects)
        else:
            st.error("Something went wrong")


def render_pages_menu(
    select_project: Callable[[str], None], local_projects: dict[str, Project], initial_project_hash: str
):
    if not is_latest(get_state().project_paths.project_dir):
        st.error("READ ONLY MODE \n\n Changes will not be saved")

    key_path = UseState(DEFAULT_PAGE_PATH)
    items = to_items(pages)
    output_state = UseState[Optional[Tuple[PagesMenuOutputAction, Optional[str]]]](None)
    output = pages_menu(items, list(local_projects.values()), initial_project_hash)
    if output and output != output_state.value:
        output_state.set(output)
        action, payload = output
        if action == PagesMenuOutputAction.VIEW_ALL_PROJECTS:
            refresh(nuke=True)
        elif action == PagesMenuOutputAction.SELECT_PROJECT and payload:
            select_project(payload)
        elif action == PagesMenuOutputAction.SELECT_PAGE and payload and payload != key_path.value:
            key_path.set(payload.split(SEPARATOR))

    return key_path.value


def main(target: str):
    set_page_config()
    target_path = Path(target)
    initial_project = UseState[Optional[Project]](None, "INITIAL_PROJECT")
    memoized_projects, refresh_projects = use_memo(lambda: get_projects(target_path))

    def select_project(project_hash: str, refetch=False):
        projects = refresh_projects() if refetch else memoized_projects
        project = {**projects.local, **projects.sandbox}.get(project_hash)
        if not project:
            return
        project_path = projects.local_paths.get(project["hash"])
        if not project_path:
            return

        if not initial_project.value:
            initial_project.set(project)

        project_dir = project_path.expanduser().absolute()
        State.init(project_dir)
        refresh(clear_memo=True)

    if not has_state() or not initial_project.value:
        render_projects_page(
            select_project=select_project,
            projects=memoized_projects,
            download_path=target_path,
        )
        return
    else:
        DBConnection.set_project_path(get_state().project_paths.project_dir)

    with st.sidebar:
        render_help()
        key_path = render_pages_menu(select_project, memoized_projects.local, initial_project.value["hash"])

    provide_backcompatibility_for_old_predictions()

    render = reduce(dict.__getitem__, key_path, pages)
    if callable(render):
        render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Directory in which your project data is stored")
    args = parser.parse_args()
    main(args.project_dir)
