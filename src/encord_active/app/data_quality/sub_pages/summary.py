import pandas as pd
import streamlit as st

from encord_active.app.common.components.metric_summary import render_metric_summary
from encord_active.app.common.components.data_quality_summary import summary_item
from encord_active.app.common.components.tags.tag_creator import tag_creator
from encord_active.lib.charts.data_quality_summary import create_outlier_distribution_chart, \
    create_image_size_distribution_chart
from encord_active.app.common.page import Page
from encord_active.app.common.state import get_state
from encord_active.lib.dataset.outliers import get_iqr_outliers, MetricWithDistanceSchema, OutlierStatus
from encord_active.lib.metrics.utils import (
    MetricScope,
    load_available_metrics,
    load_metric_dataframe,
)

from encord_active.lib.dataset.summary_utils import get_all_image_sizes, get_median_value_of_2D_array

_COLUMNS = MetricWithDistanceSchema


class SummaryPage(Page):
    title = "📑 Summary"
    _severe_outlier_color = 'tomato'
    _moderate_outlier_color = 'orange'

    def sidebar_options(self):
        self.row_col_settings_in_sidebar()
        tag_creator()

    def build(self, metric_scope: MetricScope):

        image_sizes = get_all_image_sizes(get_state().project_paths.project_dir)
        median_dimension = get_median_value_of_2D_array(image_sizes)

        metrics = load_available_metrics(get_state().project_paths.metrics, metric_scope)
        total_severe_outliers = set()
        total_moderate_outliers = set()

        all_metrics_plotting = pd.DataFrame(columns=['metric', 'total_severe_outliers', 'total_moderate_outliers'])

        for metric in metrics:
            original_df = load_metric_dataframe(metric, normalize=False)
            res = get_iqr_outliers(original_df)
            if not res:
                continue

            df, iqr_outliers = res

            all_metrics_plotting = pd.concat([all_metrics_plotting, pd.DataFrame(
                {'metric': [metric.name], 'total_severe_outliers': [iqr_outliers.n_severe_outliers],
                 'total_moderate_outliers': [iqr_outliers.n_moderate_outliers]})], axis=0)

            for _, row in df.iterrows():
                if row[_COLUMNS.outliers_status] == OutlierStatus.severe.value:
                    total_severe_outliers.add(row[_COLUMNS.identifier])
                elif row[_COLUMNS.outliers_status] == OutlierStatus.moderate.value:
                    total_moderate_outliers.add(row[_COLUMNS.identifier])

        all_metrics_plotting.sort_values(by=['total_severe_outliers'], ascending=False, inplace=True)

        st.markdown(f"# {self.title}")
        total_images_col, total_severe_outliers_col, total_moderate_outliers_col, average_image_size = st.columns(4)

        summary_item(total_images_col, 'Number of images', len(image_sizes), background_color='#f2f2f2')
        summary_item(total_severe_outliers_col, 'Total severe outliers', len(total_severe_outliers),
                     background_color=self._severe_outlier_color)
        summary_item(total_moderate_outliers_col, 'Total moderate outliers', len(total_moderate_outliers),
                     background_color=self._moderate_outlier_color)
        summary_item(average_image_size, 'Median image size', f'{median_dimension[0]}x{median_dimension[1]}',
                     background_color='#e6e6ff')

        st.write('')
        outliers_plotting_col, issues_col = st.columns([6, 3])

        # Outlier Distributions
        fig = create_outlier_distribution_chart(all_metrics_plotting, self._severe_outlier_color,
                                                self._moderate_outlier_color)
        outliers_plotting_col.plotly_chart(fig, use_container_width=True)

        # Image size distribution
        fig = create_image_size_distribution_chart(image_sizes)
        outliers_plotting_col.plotly_chart(fig, use_container_width=True)

        metrics_with_severe_outliers = all_metrics_plotting[all_metrics_plotting['total_severe_outliers'] > 0]
        issues_col.subheader(f'You have {metrics_with_severe_outliers.shape[0]} issues to fix in your dataset')

        for _, row in metrics_with_severe_outliers.iterrows():
            summary_item(issues_col, f"{row['metric']} outliers", row['total_severe_outliers'], '#ffcc99',
                         value_html_tag='h3', value_margin_bottom='-10')
            issues_col.write('')

        with st.expander("Outlier description", expanded=True):
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
