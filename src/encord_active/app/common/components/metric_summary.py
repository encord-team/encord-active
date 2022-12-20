from typing import List

import pandas as pd
import streamlit as st
from pandera.typing import DataFrame

import encord_active.app.common.state as state
from encord_active.app.common.components import build_data_tags
from encord_active.app.common.components.individual_tagging import multiselect_tag
from encord_active.lib.common.image_utils import show_image_and_draw_polygons
from encord_active.lib.metrics.load_metrics import MetricData
from encord_active.lib.metrics.outliers import IqrOutliers, MetricWithDistanceSchema

_COLUMNS = MetricWithDistanceSchema


def render_metric_summary(metric: MetricData, df: DataFrame[MetricWithDistanceSchema], iqr_outliers: IqrOutliers):
    n_cols = int(st.session_state[state.MAIN_VIEW_COLUMN_NUM])
    n_rows = int(st.session_state[state.MAIN_VIEW_ROW_NUM])
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
            render_summary_item(row, metric.name, iqr_outliers)


def render_summary_item(row, metric_name: str, iqr_outliers: IqrOutliers):
    image = show_image_and_draw_polygons(row)
    st.image(image)

    multiselect_tag(row, f"{metric_name}_summary")

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
