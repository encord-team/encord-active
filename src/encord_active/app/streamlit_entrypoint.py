import argparse
import shutil
from pathlib import Path
from typing import Optional

import streamlit as st
from encord_active_components.components.projects_page import Project

from encord_active.app.common.components.help.help import render_help
from encord_active.app.common.state import State, get_state, has_state, refresh
from encord_active.app.common.state_hooks import UseState, use_memo
from encord_active.app.common.utils import set_page_config
from encord_active.app.page_menu import render_pages_menu
from encord_active.app.projects_page import get_projects, render_projects_page
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.model_predictions.writer import MainPredictionType


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


def main(target: str):
    set_page_config()
    target_path = Path(target)
    initial_project = UseState[Optional[Project]](None, "INITIAL_PROJECT")
    memoized_projects, refresh_projects = use_memo(lambda: get_projects(target_path))

    def select_project(project_hash: str, refetch=False):
        projects, local_paths = refresh_projects() if refetch else memoized_projects
        project = projects.get(project_hash)
        if not project:
            return
        project_path = local_paths.get(project["hash"])
        if not project_path:
            return

        if not initial_project.value:
            initial_project.set(project)

        project_dir = project_path.expanduser().absolute()
        State.init(project_dir, refresh_projects)
        refresh(clear_memo=True)

    if not has_state() or not initial_project.value:
        render_projects_page(
            select_project=select_project,
            projects=memoized_projects.projects,
            download_path=target_path,
        )
        return
    else:
        DBConnection.set_project_path(get_state().project_paths.project_dir)

    with st.sidebar:
        render_help()
        try:
            render = render_pages_menu(select_project, memoized_projects.projects, initial_project.value["hash"])
        except Exception as e:
            __import__("ipdb").set_trace()
            pass

    provide_backcompatibility_for_old_predictions()

    if callable(render):
        render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Directory in which your project data is stored")
    args = parser.parse_args()
    main(args.project_dir)
