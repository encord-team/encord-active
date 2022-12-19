import streamlit as st

from encord_active.app.common.components import sticky_header
from encord_active.app.common.utils import setup_page
from encord_active.app.data_quality.sub_pages.explorer import ExplorerPage
from encord_active.app.data_quality.sub_pages.summary import SummaryPage
from encord_active.lib.metrics.load_metrics import MetricScope, load_available_metrics


def summary(metric_type: MetricScope):
    def render():
        setup_page()
        page = SummaryPage()

        with sticky_header():
            page.sidebar_options()

        page.build(metric_type)

    return render


def explorer(metric_type: MetricScope):
    def render():
        setup_page()
        page = ExplorerPage()
        available_metrics = load_available_metrics(st.session_state.metric_dir, metric_type)

        with sticky_header():
            selected_df = page.sidebar_options(available_metrics)

        if selected_df is None:
            return

        page.build(selected_df)

    return render
