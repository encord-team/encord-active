import argparse
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import streamlit as st
from encord_active_components import MenuItem, pages_menu

from encord_active.app.actions_page.export_balance import export_balance
from encord_active.app.actions_page.export_filter import export_filter
from encord_active.app.common.components.help.help import render_help
from encord_active.app.common.state import State
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

Pages = Dict[str, Union[Callable, "Pages"]]  # type: ignore

pages: Pages = {
    "Data Quality": {"Summary": summary(MetricScope.DATA_QUALITY), "Explorer": explorer(MetricScope.DATA_QUALITY)},
    "Label Quality": {"Summary": summary(MetricScope.LABEL_QUALITY), "Explorer": explorer(MetricScope.LABEL_QUALITY)},
    "Model Quality": {
        "Metrics": model_quality(MetricsPage()),
        "Performance By Metric": model_quality(PerformanceMetric()),
        "True Positives": model_quality(TruePositivesPage()),
        "False Positives": model_quality(FalsePositivesPage()),
        "False Negatives": model_quality(FalseNegativesPage()),
    },
    "Actions": {"Filter & Export": export_filter, "Balance & Export": export_balance},
}

SEPARATOR = "#"


def to_item(k, v, parent_key: Optional[str] = None):
    # NOTE: keys must be unique for the menu to render properly
    composite_key = SEPARATOR.join(filter(None, [parent_key, k]))
    item = MenuItem(key=composite_key, label=k, children=None)
    if isinstance(v, dict):
        item["children"] = to_items(v, parent_key=composite_key)
    return item


def to_items(d: dict, parent_key: Optional[str] = None):
    return [to_item(k, v, parent_key) for k, v in d.items()]


def main(target: str):
    set_page_config()
    render_help()
    target_path = Path(target)

    if is_project(target_path):
        projects = [target_path]
    else:
        parent_project = try_find_parent_project(target_path)
        if parent_project:
            projects = [parent_project]
        else:
            projects = find_child_projects(target_path)

    with st.sidebar:
        project_path = st.selectbox("Choose Project", projects, format_func=lambda path: path.name)
        if project_path:
            items = to_items(pages)
            key = pages_menu(items)
            path = key.split(SEPARATOR) if key else []

    if not project_path:
        st.info("Please choose a project")
        return

    project_dir = Path(project_path).expanduser().absolute()
    st.session_state.project_dir = project_dir

    if not st.session_state.project_dir.is_dir():
        st.error(f"Project not found for directory `{project_path}`.")
        st.stop()

    DBConnection.set_project_path(project_dir)
    State.init(project_dir)

    render = reduce(dict.__getitem__, path, pages) if path else pages["Data Quality"]["Summary"]  # type: ignore
    if callable(render):
        render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Directory in which your project data is stored")
    args = parser.parse_args()
    main(args.project_dir)
