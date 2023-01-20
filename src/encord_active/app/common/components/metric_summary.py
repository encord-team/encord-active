from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from pandera.typing import DataFrame

from encord_active.app.common.components import build_data_tags
from encord_active.app.common.components.data_quality_summary import summary_item
from encord_active.app.common.components.tags.individual_tagging import multiselect_tag
from encord_active.app.common.state import get_state
from encord_active.lib.common.image_utils import show_image_and_draw_polygons
from encord_active.lib.dataset.outliers import IqrOutliers, MetricWithDistanceSchema
from encord_active.lib.metrics.utils import MetricData, MetricScope

_COLUMNS = MetricWithDistanceSchema


def render_data_quality_summary_top_bar(median_image_dimension: np.ndarray, background_color: str):
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


def render_label_quality_summary_top_bar(background_color: str):
    (
        total_object_annotations_col,
        total_classification_annotations_col,
        total_severe_outliers_col,
        total_moderate_outliers_col,
    ) = st.columns(4)

    total_object_annotations_col.markdown(
        summary_item("Object annotations", get_state().annotation_sizes[1], background_color=background_color),
        unsafe_allow_html=True,
    )

    total_classification_annotations_col.markdown(
        summary_item(
            "Classification annotations",
            get_state().annotation_sizes[0],
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
