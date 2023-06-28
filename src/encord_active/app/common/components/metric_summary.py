from typing import List, cast

import pandas as pd
import streamlit as st
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator

from encord_active.app.common.components.data_quality_summary import summary_item
from encord_active.app.common.state import get_state
from encord_active.app.common.state_hooks import use_memo
from encord_active.lib.charts.data_quality_summary import (
    CrossMetricSchema,
    create_2d_metric_chart,
    create_image_size_distribution_chart,
    create_labels_distribution_chart,
    create_outlier_distribution_chart,
)
from encord_active.lib.dataset.outliers import (
    AllMetricsOutlierSchema,
    MetricsSeverity,
    MetricWithDistanceSchema,
    get_all_metrics_outliers,
)
from encord_active.lib.dataset.summary_utils import (
    get_all_annotation_numbers,
    get_all_image_sizes,
    get_median_value_of_2d_array,
    get_metric_summary,
)
from encord_active.lib.metrics.utils import MetricData, MetricScope


def render_2d_metric_plots(metrics_data_summary: MetricsSeverity):
    with st.expander("2D metrics view", True):
        metric_selection_col, scatter_plot_col = st.columns([2, 5])

        # Annotation Duplicates has different identifier structures than the other metrics: therefore, it is
        # excluded from the 2D metric view for now.
        metric_names = sorted(
            [metric_key for metric_key in metrics_data_summary.metrics.keys() if metric_key != "Annotation Duplicates"]
        )

        if len(metric_names) < 2:
            st.info("You need at least two metrics to plot 2D metric view.")
            return

        x_metric_name = metric_selection_col.selectbox("x axis", metric_names, index=0)
        y_metric_name = metric_selection_col.selectbox("y axis", metric_names, index=1)
        trend_selected = metric_selection_col.checkbox(
            "Show trend",
            value=True,
            help="Draws a trend line to demonstrate the relationship between the two metrics.",
        )

        x_metric_df = (
            metrics_data_summary.metrics[str(x_metric_name)]
            .df[[MetricWithDistanceSchema.identifier, MetricWithDistanceSchema.score]]
            .copy()
        )
        x_metric_df.rename(columns={MetricWithDistanceSchema.score: f"{CrossMetricSchema.x}"}, inplace=True)

        y_metric_df = (
            metrics_data_summary.metrics[str(y_metric_name)]
            .df[[MetricWithDistanceSchema.identifier, MetricWithDistanceSchema.score]]
            .copy()
        )
        y_metric_df.rename(columns={MetricWithDistanceSchema.score: f"{CrossMetricSchema.y}"}, inplace=True)

        if x_metric_df.shape[0] == 0:
            st.info(f'Score file of metric "{x_metric_name}" is empty, please run this metric again.')
            return

        if y_metric_df.shape[0] == 0:
            st.info(f'Score file of metric "{y_metric_name}" is empty, please run this metric again.')
            return

        if len(x_metric_df.iloc[0][MetricWithDistanceSchema.identifier].split("_")) == len(
            y_metric_df.iloc[0][MetricWithDistanceSchema.identifier].split("_")
        ):
            merged_metrics = pd.merge(x_metric_df, y_metric_df, how="inner", on=MetricWithDistanceSchema.identifier)
        else:
            x_changed, to_be_parsed_df = (
                (True, x_metric_df.copy(deep=True))
                if len(x_metric_df.iloc[0][MetricWithDistanceSchema.identifier].split("_")) == 4
                else (False, y_metric_df.copy(deep=True))
            )

            to_be_parsed_df[[MetricWithDistanceSchema.identifier, "identifier_rest"]] = to_be_parsed_df[
                MetricWithDistanceSchema.identifier
            ].str.rsplit("_", n=1, expand=True)

            merged_metrics = pd.merge(
                to_be_parsed_df if x_changed else x_metric_df,
                y_metric_df if x_changed else to_be_parsed_df,
                how="inner",
                on=MetricWithDistanceSchema.identifier,
            )
            merged_metrics[MetricWithDistanceSchema.identifier] = (
                merged_metrics[MetricWithDistanceSchema.identifier] + "_" + merged_metrics["identifier_rest"]
            )

            merged_metrics.pop("identifier_rest")
        if not merged_metrics.empty:
            fig = create_2d_metric_chart(
                merged_metrics.pipe(DataFrame[CrossMetricSchema]),
                str(x_metric_name),
                str(y_metric_name),
                trend_selected,
            )
            scatter_plot_col.plotly_chart(fig, use_container_width=True)


def render_issues_pane(metrics: DataFrame[AllMetricsOutlierSchema], st_col: DeltaGenerator, metric_scope: MetricScope):
    if metric_scope == MetricScope.DATA_QUALITY:
        issue_context = "data"
    elif metric_scope == MetricScope.LABEL_QUALITY:
        issue_context = "labels"
    else:
        issue_context = "project"

    st_col.subheader(f":triangular_flag_on_post: {metrics.shape[0]} issues to fix in your {issue_context}")

    for counter, (_, row) in enumerate(metrics.iterrows()):
        st_col.metric(
            label=f"**{counter + 1}**\. {row[AllMetricsOutlierSchema.metric_name]}",
            value=row[AllMetricsOutlierSchema.total_severe_outliers],
            help=f"Go to Explorer page and chose {row[AllMetricsOutlierSchema.metric_name]} metric to spot these outliers.",
        )


def render_data_quality_dashboard(
    metrics: List[MetricData], severe_outlier_color: str, moderate_outlier_color: str, background_color: str
):
    image_sizes, _ = use_memo(lambda: get_all_image_sizes(get_state().project_paths))
    median_image_dimension = get_median_value_of_2d_array(image_sizes)

    if get_state().metrics_data_summary is None:
        get_state().metrics_data_summary = get_metric_summary(metrics)

    all_metrics_outliers = get_all_metrics_outliers(get_state().metrics_data_summary)

    total_images_col, total_severe_outliers_col, total_moderate_outliers_col, average_image_size = st.columns(4)

    total_images_col.markdown(
        summary_item("Number of images", len(image_sizes), background_color=background_color),
        unsafe_allow_html=True,
    )

    total_severe_outliers_col.markdown(
        summary_item(
            "🔴 Total severe outliers",
            get_state().metrics_data_summary.total_unique_severe_outliers,
            background_color=background_color,
        ),
        unsafe_allow_html=True,
    )

    total_moderate_outliers_col.markdown(
        summary_item(
            "🟠 Total moderate outliers",
            get_state().metrics_data_summary.total_unique_moderate_outliers,
            background_color=background_color,
        ),
        unsafe_allow_html=True,
    )

    average_image_size.markdown(
        summary_item(
            "Median image size",
            f"{median_image_dimension[0]}x{median_image_dimension[1]}",
            background_color=background_color,
        ),
        unsafe_allow_html=True,
    )

    st.write("")
    plots_col, issues_col = st.columns([6, 3])

    if get_state().metrics_data_summary.total_unique_severe_outliers > 0:
        fig = create_outlier_distribution_chart(all_metrics_outliers, severe_outlier_color, moderate_outlier_color)
        plots_col.plotly_chart(fig, use_container_width=True)

    fig = create_image_size_distribution_chart(image_sizes)
    plots_col.plotly_chart(fig, use_container_width=True)

    metrics_with_severe_outliers = all_metrics_outliers[
        all_metrics_outliers[AllMetricsOutlierSchema.total_severe_outliers] > 0
    ]
    render_issues_pane(
        cast(DataFrame[AllMetricsOutlierSchema], metrics_with_severe_outliers), issues_col, MetricScope.DATA_QUALITY
    )

    render_2d_metric_plots(get_state().metrics_data_summary)


def render_label_quality_dashboard(
    metrics: List[MetricData], severe_outlier_color: str, moderate_outlier_color: str, background_color: str
):
    if get_state().annotation_sizes is None:
        get_state().annotation_sizes = get_all_annotation_numbers(get_state().project_paths)

    if get_state().metrics_label_summary is None:
        get_state().metrics_label_summary = get_metric_summary(metrics)

    all_metrics_outliers = get_all_metrics_outliers(get_state().metrics_label_summary)

    (
        total_object_annotations_col,
        total_classification_annotations_col,
        total_severe_outliers_col,
        total_moderate_outliers_col,
    ) = st.columns(4)

    total_object_annotations_col.markdown(
        summary_item(
            "Object annotations", get_state().annotation_sizes.total_object_labels, background_color=background_color
        ),
        unsafe_allow_html=True,
    )

    total_classification_annotations_col.markdown(
        summary_item(
            "Classification annotations",
            get_state().annotation_sizes.total_classification_labels,
            background_color=background_color,
        ),
        unsafe_allow_html=True,
    )

    total_severe_outliers_col.markdown(
        summary_item(
            "🔴 Total severe outliers",
            get_state().metrics_label_summary.total_unique_severe_outliers,
            background_color=background_color,
        ),
        unsafe_allow_html=True,
    )

    total_moderate_outliers_col.markdown(
        summary_item(
            "🟠 Total moderate outliers",
            get_state().metrics_label_summary.total_unique_moderate_outliers,
            background_color=background_color,
        ),
        unsafe_allow_html=True,
    )

    st.write("")
    plots_col, issues_col = st.columns([6, 3])

    if get_state().metrics_label_summary.total_unique_severe_outliers > 0:
        fig = create_outlier_distribution_chart(all_metrics_outliers, severe_outlier_color, moderate_outlier_color)
        plots_col.plotly_chart(fig, use_container_width=True)

    # label distribution plots
    with plots_col.expander("Labels distribution", expanded=True):
        if (
            get_state().annotation_sizes.total_object_labels > 0
            or get_state().annotation_sizes.total_classification_labels > 0
        ):
            st.info("If a class's size is lower than half of the median value, it is indicated as 'undersampled'.")

        if get_state().annotation_sizes.total_object_labels > 0:
            fig = create_labels_distribution_chart(
                get_state().annotation_sizes.objects, "Objects distributions", "Object"
            )
            st.plotly_chart(fig, use_container_width=True)

        for (
            classification_question_name,
            classification_question_answers,
        ) in get_state().annotation_sizes.classifications.items():
            if classification_question_answers:
                fig = create_labels_distribution_chart(
                    classification_question_answers, classification_question_name, "Class"
                )
                st.plotly_chart(fig, use_container_width=True)

    metrics_with_severe_outliers = all_metrics_outliers[
        all_metrics_outliers[AllMetricsOutlierSchema.total_severe_outliers] > 0
    ]
    render_issues_pane(
        cast(DataFrame[AllMetricsOutlierSchema], metrics_with_severe_outliers), issues_col, MetricScope.LABEL_QUALITY
    )
    render_2d_metric_plots(get_state().metrics_label_summary)
