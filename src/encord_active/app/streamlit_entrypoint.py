import argparse
from functools import reduce
from typing import Callable, Dict, Optional, Union

import streamlit as st

from encord_active.app.actions_page.export_balance import export_balance
from encord_active.app.actions_page.export_filter import export_filter
from encord_active.app.common import state
from encord_active.app.common.utils import set_page_config
from encord_active.app.data_quality.common import MetricType
from encord_active.app.frontend_components import pages_menu
from encord_active.app.model_assertions.sub_pages.false_negatives import (
    FalseNegativesPage,
)
from encord_active.app.model_assertions.sub_pages.false_positives import (
    FalsePositivesPage,
)
from encord_active.app.model_assertions.sub_pages.metrics import MetricsPage
from encord_active.app.model_assertions.sub_pages.performance_by_metric import (
    PerformanceMetric,
)
from encord_active.app.model_assertions.sub_pages.true_positives import (
    TruePositivesPage,
)
from encord_active.app.views.landing_page import landing_page
from encord_active.app.views.metrics import explorer, summary
from encord_active.app.views.model_assertions import model_assertions

Pages = Dict[str, Union[Callable, "Pages"]]  # type: ignore

pages: Pages = {
    "Encord Active": landing_page,
    "Data Quality": {"Summary": summary(MetricType.DATA_QUALITY), "Explorer": explorer(MetricType.DATA_QUALITY)},
    "Label Quality": {"Summary": summary(MetricType.LABEL_QUALITY), "Explorer": explorer(MetricType.LABEL_QUALITY)},
    "Model Assertions": {
        "Metrics": model_assertions(MetricsPage()),
        "Performance By Metric": model_assertions(PerformanceMetric()),
        "True Positives": model_assertions(TruePositivesPage()),
        "False Positives": model_assertions(FalsePositivesPage()),
        "False Negatives": model_assertions(FalseNegativesPage()),
    },
    "Actions": {"Filter & Export": export_filter, "Balance & Export": export_balance},
}

SEPARATOR = "#"


def to_item(k, v, parent_key: Optional[str] = None):
    # NOTE: keys must be unique for the menu to render properly
    composite_key = SEPARATOR.join(filter(None, [parent_key, k]))
    item = {"key": composite_key, "label": k}
    if isinstance(v, dict):
        item["children"] = to_items(v, parent_key=composite_key)
    return item


def to_items(d: dict, parent_key: Optional[str] = None):
    return [to_item(k, v, parent_key) for k, v in d.items()]


def main(project_path: str):
    set_page_config()

    if not state.set_project_dir(project_path):
        st.error(f"Project not found for directory `{project_path}`.")

    with st.sidebar:
        items = to_items(pages)
        key = pages_menu(items=items)
        path = key.split(SEPARATOR) if key else []

    render = reduce(dict.__getitem__, path, pages) if path else pages["Encord Active"]  # type: ignore
    if callable(render):
        render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=str, help="Directory in which your project data is stored")
    args = parser.parse_args()
    main(args.project_dir)
