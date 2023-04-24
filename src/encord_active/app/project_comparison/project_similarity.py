import streamlit as st

from encord_active.app.common.state import get_state
from encord_active.app.common.utils import setup_page
from encord_active.lib.metrics.utils import MetricScope, load_available_metrics


def project_similarity():
    def render():
        setup_page()

        st.write("Hello")

    return render
