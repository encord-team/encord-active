from abc import abstractmethod

import altair as alt
import pandas as pd
import streamlit as st

import encord_active.app.common.state as state
from encord_active.app.common.page import Page
from encord_active.lib.metrics.statistical_utils import get_histogram


class ModelQualityPage(Page):
    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @abstractmethod
    def sidebar_options(self):
        """
        Used to append options to the sidebar.
        """
        pass

    @abstractmethod
    def build(
        self,
        model_predictions: pd.DataFrame,
        labels: pd.DataFrame,
        metrics: pd.DataFrame,
        precisions: pd.DataFrame,
    ):
        pass

    def __call__(
        self,
        model_predictions: pd.DataFrame,
        labels: pd.DataFrame,
        metrics: pd.DataFrame,
        precisions: pd.DataFrame,
    ):
        return self.build(model_predictions, labels, metrics, precisions)

    def __repr__(self):
        return f"{type(self).__name__}()"

    @staticmethod
    def prediction_metric_in_sidebar():
        """
        Note: Adding the fixed options "confidence" and "iou" works here because
        confidence is required on import and IOU is computed during prediction
        import. So the two columns are already available in the
        `st.session_state.model_predictions` data frame.
        """
        fixed_options = {"confidence": "Model Confidence", "iou": "IOU"}
        st.selectbox(
            "Select metric for your predictions",
            st.session_state.prediction_metric_names + list(fixed_options.keys()),
            key=state.PREDICTIONS_METRIC,
            format_func=lambda s: fixed_options.get(s, s),
            help="The data in the main view will be sorted by the selected metric. "
            "(F) := frame scores, (P) := prediction scores.",
        )

    @staticmethod
    def metric_details_description(metric_name: str = ""):
        if not metric_name:
            metric_name = st.session_state[state.PREDICTIONS_METRIC]
        metric_meta = st.session_state.metric_meta["prediction"].get(metric_name[:-4], {})  # Remove " (P)"
        if not metric_meta:
            metric_meta = st.session_state.metric_meta["data"].get(metric_name[:-4], {})  # Remove " (P)"
        if metric_meta:
            st.markdown(f"### The {metric_meta['title']} metric")
            st.markdown(metric_meta["long_description"])
