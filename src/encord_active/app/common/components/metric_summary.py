from typing import List

import pandas as pd
import streamlit as st
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator

from encord_active.app.common.components import build_data_tags
from encord_active.app.common.components.data_quality_summary import summary_item
from encord_active.app.common.components.tags.individual_tagging import multiselect_tag
from encord_active.app.common.state import MetricOutlierInfo, MetricsSeverity, get_state
from encord_active.lib.charts.data_quality_summary import (
    create_image_size_distribution_chart,
    create_labels_distribution_chart,
    create_outlier_distribution_chart,
)
from encord_active.lib.common.image_utils import show_image_and_draw_polygons
from encord_active.lib.dataset.outliers import (
    IqrOutliers,
    MetricWithDistanceSchema,
    Severity,
    get_iqr_outliers,
)
from encord_active.lib.dataset.summary_utils import (
    get_all_annotation_numbers,
    get_all_image_sizes,
    get_median_value_of_2d_array,
)
from encord_active.lib.metrics.utils import (
    MetricData,
    MetricScope,
    load_available_metrics,
    load_metric_dataframe,
)

_COLUMNS = MetricWithDistanceSchema


def get_metric_summary(metrics: list[MetricData]) -> MetricsSeverity:
    metric_severity = MetricsSeverity()
    total_unique_severe_outliers = set()
    total_unique_moderate_outliers = set()

    for metric in metrics:
        original_df = load_metric_dataframe(metric, normalize=False)
        res = get_iqr_outliers(original_df)

        if not res:
            continue

        df, iqr_outliers = res

        df_dict = df.to_dict("records")
        for row in df_dict:
            if row[_COLUMNS.outliers_status] == Severity.severe:
                total_unique_severe_outliers.add(row[_COLUMNS.identifier])
            elif row[_COLUMNS.outliers_status] == Severity.moderate:
                total_unique_moderate_outliers.add(row[_COLUMNS.identifier])

        metric_severity.metrics.append(MetricOutlierInfo(metric=metric, df=df, iqr_outliers=iqr_outliers))

    metric_severity.total_unique_severe_outliers = len(total_unique_severe_outliers)
    metric_severity.total_unique_moderate_outliers = len(total_unique_moderate_outliers)

    return metric_severity


def get_all_metrics_outliers(metrics_data_summary: MetricsSeverity) -> pd.DataFrame:
    all_metrics_outliers = pd.DataFrame(columns=["metric", "total_severe_outliers", "total_moderate_outliers"])
    for item in metrics_data_summary.metrics:
        all_metrics_outliers = pd.concat(
            [
                all_metrics_outliers,
                pd.DataFrame(
                    {
                        "metric": [item.metric.name],
                        "total_severe_outliers": [item.iqr_outliers.n_severe_outliers],
                        "total_moderate_outliers": [item.iqr_outliers.n_moderate_outliers],
                    }
                ),
            ],
            axis=0,
        )

    all_metrics_outliers.sort_values(by=["total_severe_outliers"], ascending=False, inplace=True)

    return all_metrics_outliers


def render_issues_pane(metrics: pd.DataFrame, st_col: DeltaGenerator):
    st_col.subheader(f":triangular_flag_on_post: {metrics.shape[0]} issues to fix in your dataset")

    for counter, (_, row) in enumerate(metrics.iterrows()):
        st_col.metric(
            f"{counter + 1}. {row['metric']} outliers",
            row["total_severe_outliers"],
            help=f'Go to Explorer page and chose {row["metric"]} metric to spot these outliers.',
        )


def render_data_quality_dashboard(severe_outlier_color: str, moderate_outlier_color: str, background_color: str):
    if get_state().image_sizes is None:
        get_state().image_sizes = get_all_image_sizes(get_state().project_paths.project_dir)
    median_image_dimension = get_median_value_of_2d_array(get_state().image_sizes)

    metrics = load_available_metrics(get_state().project_paths.metrics, MetricScope.DATA_QUALITY)
    if get_state().metrics_data_summary is None:
        get_state().metrics_data_summary = get_metric_summary(metrics)

    all_metrics_outliers = get_all_metrics_outliers(get_state().metrics_data_summary)

    total_images_col, total_severe_outliers_col, total_moderate_outliers_col, average_image_size = st.columns(4)

    total_images_col.markdown(
        summary_item("Number of images", len(get_state().image_sizes), background_color=background_color),
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

    fig = create_image_size_distribution_chart(get_state().image_sizes)
    plots_col.plotly_chart(fig, use_container_width=True)

    metrics_with_severe_outliers = all_metrics_outliers[all_metrics_outliers["total_severe_outliers"] > 0]
    render_issues_pane(metrics_with_severe_outliers, issues_col)


def render_label_quality_dashboard(severe_outlier_color: str, moderate_outlier_color: str, background_color: str):

    if get_state().annotation_sizes is None:
        get_state().annotation_sizes = get_all_annotation_numbers(get_state().project_paths.project_dir)

    metrics = load_available_metrics(get_state().project_paths.metrics, MetricScope.LABEL_QUALITY)
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

        for classification_question_name, classification_question_answers in (
            get_state().annotation_sizes.classifications.items()
        ):
            fig = create_labels_distribution_chart(
                classification_question_answers, classification_question_name, "Class"
            )
            st.plotly_chart(fig, use_container_width=True)

    metrics_with_severe_outliers = all_metrics_outliers[all_metrics_outliers["total_severe_outliers"] > 0]
    render_issues_pane(metrics_with_severe_outliers, issues_col)


def render_metric_summary(
    metric: MetricData, df: DataFrame[MetricWithDistanceSchema], iqr_outliers: IqrOutliers, metric_scope: MetricScope
):
    n_cols = get_state().page_grid_settings.columns
    n_rows = get_state().page_grid_settings.rows
    page_size = n_cols * n_rows

    st.markdown(metric.meta["long_description"])

    if iqr_outliers.n_severe_outliers + iqr_outliers.n_moderate_outliers == 0:
        st.success("No outliers found!")
        return None

    st.error(f"Number of severe outliers: {iqr_outliers.n_severe_outliers}/{len(df)}")
    st.warning(f"Number of moderate outliers: {iqr_outliers.n_moderate_outliers}/{len(df)}")

    max_value = float(df[_COLUMNS.dist_to_iqr].max())
    min_value = float(df[_COLUMNS.dist_to_iqr].min())
    value = st.slider(
        "distance to IQR",
        min_value=min_value,
        max_value=max_value,
        step=max(0.1, float((max_value - min_value) / (len(df) / page_size))),
        value=max_value,
        key=f"dist_to_iqr{metric.name}",
    )

    selected_df: DataFrame[MetricWithDistanceSchema] = df[df[_COLUMNS.dist_to_iqr] <= value][:page_size]

    cols: List = []
    for _, row in selected_df.iterrows():
        if not cols:
            cols = list(st.columns(n_cols))

        with cols.pop(0):
            render_summary_item(row, metric.name, iqr_outliers, metric_scope)


def render_summary_item(row, metric_name: str, iqr_outliers: IqrOutliers, metric_scope: MetricScope):
    image = show_image_and_draw_polygons(row, get_state().project_paths.data)
    st.image(image)

    multiselect_tag(row, f"{metric_name}_summary", metric_scope)

    # === Write scores and link to editor === #

    tags_row = row.copy()
    if row["score"] > iqr_outliers.severe_ub or row["score"] < iqr_outliers.severe_lb:
        tags_row["outlier"] = "Severe"
    elif row["score"] > iqr_outliers.moderate_ub or row["score"] < iqr_outliers.moderate_lb:
        tags_row["outlier"] = "Moderate"
    else:
        tags_row["outlier"] = "Low"

    if "object_class" in tags_row and not pd.isna(tags_row["object_class"]):
        tags_row["label_class_name"] = tags_row["object_class"]
        tags_row.drop("object_class")
    tags_row[metric_name] = tags_row["score"]
    build_data_tags(tags_row, metric_name)

    if not pd.isnull(row["description"]):
        st.write(f"Description: {row['description']}. ")
