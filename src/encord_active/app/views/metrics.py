import streamlit as st
from encord_active_components.components.active import active

from encord_active.app.common.state import get_state
from encord_active.app.common.utils import setup_page
from encord_active.app.data_quality.sub_pages.explorer import ExplorerPage
from encord_active.app.data_quality.sub_pages.summary import SummaryPage
from encord_active.app.label_onboarding.label_onboarding import label_onboarding_page
from encord_active.lib.metrics.utils import MetricScope, load_available_metrics
from encord_active.server.settings import get_settings


def summary(metric_type: MetricScope):
    def render():
        setup_page()
        page = SummaryPage()

        available_metrics = load_available_metrics(get_state().project_paths.metrics, metric_type)
        if not available_metrics:
            if metric_type == MetricScope.LABEL_QUALITY:
                return label_onboarding_page()
            else:
                st.error("You don't seem to have any data in your project.")
                return

        page.sidebar_options(metric_type)
        project_hash = get_state().project_paths.load_project_meta().get("project_hash")
        active(project_hash=project_hash, api_url=get_settings().API_URL)

    return render


def explorer(metric_type: MetricScope):
    def render():
        setup_page()
        page = ExplorerPage()
        available_metrics = load_available_metrics(get_state().project_paths.metrics, metric_type)

        if not available_metrics:
            if metric_type == MetricScope.LABEL_QUALITY:
                return label_onboarding_page()
            else:
                st.error("You don't seem to have any data in your project.")
                return

        page.sidebar_options(available_metrics, metric_type)
        page.build(metric_type)

    return render
