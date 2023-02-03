import streamlit as st

from encord_active.app.common.components.metric_summary import (
    render_data_quality_dashboard,
    render_label_quality_dashboard,
    render_metric_summary,
)
from encord_active.app.common.components.tags.tag_creator import tag_creator
from encord_active.app.common.page import Page
from encord_active.app.common.state import get_state
from encord_active.app.label_onboarding.label_onboarding import label_onboarding_page
from encord_active.lib.dataset.outliers import MetricWithDistanceSchema
from encord_active.lib.metrics.utils import MetricScope, load_available_metrics

_COLUMNS = MetricWithDistanceSchema


class SummaryPage(Page):
    title = "📑 Summary"
    _severe_outlier_color = "tomato"
    _moderate_outlier_color = "orange"
    _summary_item_background_color = "#fbfbfb"

    def sidebar_options(self):
        self.row_col_settings_in_sidebar()
        tag_creator()

    def build(self, metric_scope: MetricScope):

        if metric_scope == MetricScope.DATA_QUALITY:
            render_data_quality_dashboard(
                self._severe_outlier_color, self._moderate_outlier_color, self._summary_item_background_color
            )
        elif metric_scope == MetricScope.LABEL_QUALITY:
            metrics = load_available_metrics(get_state().project_paths.metrics, MetricScope.LABEL_QUALITY)
            if not metrics:
                return label_onboarding_page()
            render_label_quality_dashboard(
                self._severe_outlier_color, self._moderate_outlier_color, self._summary_item_background_color
            )

        with st.expander("Outlier description", expanded=False):
            st.write(
                "Box plot visualisation is a common technique in descriptive statistics to detect outliers. "
                "Interquartile range ($IQR$) refers to the difference between the 25th ($Q1$) and 75th ($Q3$)"
                " percentile and tells how spread the middle values are."
            )
            st.write(
                "Here we uses $score$ values for each metric to determine their outlier status according to "
                "how distant "
                "they are to the minimum or maximum values:"
            )

            st.markdown(
                '- <i class="fa-solid fa-circle text-red"></i>  **Severe**: '
                "used for those samples where $(score \leq Q1-2.5 \\times IQR) \lor (Q3+2.5 \\times IQR \leq "
                "score)$. \n"
                '- <i class="fa-solid fa-circle text-orange"></i> **Moderate**: '
                "used for those samples where $score$ does not fall into **Severe** status and $(score \leq Q1-1.5 "
                "\\times IQR) \lor (Q3+1.5 \\times IQR \leq score)$. \n"
                '- <i class="fa-solid fa-circle text-green"></i> **Low**: '
                "used for those samples where $Q1-1.5 \\times IQR < score < Q3+1.5 \\times IQR$. ",
                unsafe_allow_html=True,
            )

            st.warning(
                "The outlier status is calculated for each metric separately; the same sample can be considered "
                "an outlier for one metric and a non-outlier for another."
            )

        current_metrics = (
            get_state().metrics_data_summary
            if metric_scope == MetricScope.DATA_QUALITY
            else get_state().metrics_label_summary
        )
        for metric_item in current_metrics.metrics.values():
            with st.expander(
                label=f"{metric_item.metric.name} Outliers - {metric_item.iqr_outliers.n_severe_outliers} severe, "
                f"{metric_item.iqr_outliers.n_moderate_outliers} moderate"
            ):
                render_metric_summary(metric_item.metric, metric_item.df, metric_item.iqr_outliers, metric_scope)
