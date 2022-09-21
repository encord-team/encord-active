from abc import ABC, abstractmethod

import altair as alt
import pandas as pd
import streamlit as st
from screeninfo import Monitor, ScreenInfoError, get_monitors

import encord_active.app.common.state as state
from encord_active.app.common.page import Page


class ModelAssertionsPage(Page):
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

    @staticmethod
    def row_col_settings_in_sidebar():
        with st.expander("Settings"):
            m_max = Monitor(height=0, width=0, x=0, y=0)
            try:
                monitors = get_monitors()
            except ScreenInfoError:
                monitors = []

            for m in monitors:
                if m.width * m.height > m_max.width * m_max.height:
                    m_max = m

            default_mv_column_num = (m_max.width // 250) or 4
            default_mv_row_num = (m_max.height // 250) or 4

            if state.MAIN_VIEW_COLUMN_NUM not in st.session_state:
                st.session_state[state.MAIN_VIEW_COLUMN_NUM] = default_mv_column_num

            if state.MAIN_VIEW_ROW_NUM not in st.session_state:
                st.session_state[state.MAIN_VIEW_ROW_NUM] = default_mv_row_num

            with st.form(key="settings_from"):
                setting_columns = st.columns(2)

                setting_columns[0].number_input(
                    "Columns",
                    min_value=2,
                    value=default_mv_column_num,
                    key=state.MAIN_VIEW_COLUMN_NUM,
                    help="Number of columns to show images in the main view",
                )

                setting_columns[1].number_input(
                    "Rows",
                    min_value=1,
                    value=default_mv_row_num,
                    key=state.MAIN_VIEW_ROW_NUM,
                    help="Number of rows to show images in the main view",
                )

                st.form_submit_button(label="Apply")


class HistogramMixin:
    @staticmethod
    def get_histogram(data_frame: pd.DataFrame, metric_column: str):
        title_suffix = f" - {metric_column}"
        bar_chart = (
            alt.Chart(data_frame, title=f"Data distribution{title_suffix}")
            .mark_bar()
            .encode(
                alt.X(f"{metric_column}:Q", bin=alt.Bin(maxbins=100), title=metric_column),
                alt.Y("count()", title="Num. samples"),
                tooltip=[
                    alt.Tooltip(f"{metric_column}:Q", title=metric_column, format=",.3f", bin=True),
                    alt.Tooltip("count():Q", title="Num. samples", format="d"),
                ],
            )
            .properties(height=200)
        )
        return bar_chart
