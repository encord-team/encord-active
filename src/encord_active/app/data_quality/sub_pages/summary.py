import streamlit as st

from encord_active.app.common.components.metric_summary import render_metric_summary
from encord_active.app.common.components.tags.tag_creator import tag_creator
from encord_active.app.common.page import Page
from encord_active.app.common.state import get_state
from encord_active.lib.dataset.outliers import get_iqr_outliers
from encord_active.lib.metrics.utils import (
    MetricScope,
    load_available_metrics,
    load_metric_dataframe,
)


class SummaryPage(Page):
    title = "📑 Summary"

    def sidebar_options(self):
        self.row_col_settings_in_sidebar()
        tag_creator()

    def build(self, metric_scope: MetricScope):
        st.markdown(f"# {self.title}")
        st.subheader("Outliers by IQR range for every metric")

        with st.expander("Description", expanded=True):
            st.write(
                "Box plot visualisation is a common technique in descriptive statistics to detect outliers. "
                "Interquartile range ($IQR$) refers to the difference between the 25th ($Q1$) and 75th ($Q3$)"
                " percentile and tells how spread the middle values are."
            )
            st.write(
                "Here we uses $score$ values for each metric to determine their outlier status according to how distant "
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

        metrics = load_available_metrics(get_state().project_paths.metrics, metric_scope)

        for metric in metrics:
            original_df = load_metric_dataframe(metric, normalize=False)
            res = get_iqr_outliers(original_df)
            if not res:
                continue

            df, iqr_outliers = res

            with st.expander(
                label=f"{metric.name} Outliers - {iqr_outliers.n_severe_outliers} severe, {iqr_outliers.n_moderate_outliers} moderate"
            ):
                render_metric_summary(metric, df, iqr_outliers, metric_scope)
