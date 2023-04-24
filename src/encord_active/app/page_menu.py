from functools import reduce
from typing import Any, Callable, List, Optional, Tuple

import streamlit as st
from encord_active_components.components.pages_menu import (
    MenuItem,
    OutputAction,
    Project,
    pages_menu,
)

from encord_active.app.actions_page.versioning import is_latest
from encord_active.app.common.state import get_state, refresh
from encord_active.app.common.state_hooks import UseState
from encord_active.app.model_quality.prediction_type_builder import ModelQualityPage
from encord_active.app.views.metrics import explorer, summary
from encord_active.app.views.model_quality import model_quality
from encord_active.lib.metrics.utils import MetricScope

PAGES = {
    "Data Quality": {"Summary": summary(MetricScope.DATA_QUALITY), "Explorer": explorer(MetricScope.DATA_QUALITY)},
    "Label Quality": {"Summary": summary(MetricScope.LABEL_QUALITY), "Explorer": explorer(MetricScope.LABEL_QUALITY)},
    "Model Quality": {
        "Metrics": model_quality(ModelQualityPage.METRICS),
        "Performance By Metric": model_quality(ModelQualityPage.PERFORMANCE_BY_METRIC),
        "Explorer": model_quality(ModelQualityPage.EXPLORER),
    },
}

DEFAULT_PAGE_PATH = ["Data Quality", "Summary"]
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


def get_renderer(key_path: List[str]):
    return reduce(dict.__getitem__, key_path, PAGES)


def render_pages_menu(
    select_project: Callable[[str], None],
    select_page: Callable[[Any], Any],
    projects: dict[str, Project],
    initial_project_hash: str,
    initial_key_path: List[str],
):
    if not is_latest(get_state().project_paths.project_dir):
        st.error("READ ONLY MODE \n\n Changes will not be saved")

    items = to_items(PAGES)
    output_state = UseState[Optional[Tuple[OutputAction, Optional[str]]]](None)
    output = pages_menu(items, list(projects.values()), SEPARATOR.join(initial_key_path), initial_project_hash)
    if output and output != output_state.value:
        output_state.set(output)
        action, payload = output
        if action == OutputAction.VIEW_ALL_PROJECTS:
            refresh(nuke=True)
        elif action == OutputAction.SELECT_PROJECT and payload:
            select_project(payload)
        elif action == OutputAction.SELECT_PAGE and payload:
            select_page(payload.split(SEPARATOR))
