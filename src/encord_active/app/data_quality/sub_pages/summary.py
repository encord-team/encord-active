from typing import List

import pandas as pd
import streamlit as st

import encord_active.app.common.state as state
from encord_active.app.common import metric as iutils
from encord_active.app.common.components import build_data_tags
from encord_active.app.common.components.individual_tagging import (
    multiselect_tag,
    tag_creator,
)
from encord_active.app.common.page import Page
from encord_active.app.common.utils import load_merged_df
from encord_active.app.data_quality.common import (
    MetricType,
    load_available_metrics,
    show_image_and_draw_polygons,
)


class SummaryPage(Page):
    title = "📑 Summary"

    def sidebar_options(self):
        self.row_col_settings_in_sidebar()
        load_merged_df()
        tag_creator()

    def build(self, metric_type_selection: MetricType):
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

        n_cols = int(st.session_state[state.MAIN_VIEW_COLUMN_NUM])
        n_rows = int(st.session_state[state.MAIN_VIEW_ROW_NUM])
        n_items_in_page = n_cols * n_rows

        metrics = load_available_metrics(metric_type_selection)

        for idx in metrics:
            df = iutils.load_metric(idx, normalize=False)

            current_df = df.copy()

            if df.empty:
                continue

            moderate_iqr_scale = 1.5
            severe_iqr_scale = 2.5
            Q1 = current_df["score"].quantile(0.25)
            Q3 = current_df["score"].quantile(0.75)
            IQR = Q3 - Q1

            current_df["dist_to_iqr"] = 0
            current_df.loc[current_df["score"] > Q3, "dist_to_iqr"] = (current_df["score"] - Q3).abs()
            current_df.loc[current_df["score"] < Q1, "dist_to_iqr"] = (current_df["score"] - Q1).abs()
            current_df.sort_values(by="dist_to_iqr", inplace=True, ascending=False)

            moderate_lb, moderate_ub = Q1 - moderate_iqr_scale * IQR, Q3 + moderate_iqr_scale * IQR
            severe_lb, severe_ub = Q1 - severe_iqr_scale * IQR, Q3 + severe_iqr_scale * IQR

            n_moderate_outliers = (
                ((severe_lb <= current_df["score"]) & (current_df["score"] < moderate_lb))
                | ((severe_ub >= current_df["score"]) & (current_df["score"] > moderate_ub))
            ).sum()

            n_severe_outliers = ((current_df["score"] < severe_lb) | (current_df["score"] > severe_ub)).sum()

            with st.expander(label=f"{idx.name} Outliers - {n_severe_outliers} severe, {n_moderate_outliers} moderate"):
                st.markdown(idx.meta["long_description"])

                if n_severe_outliers + n_moderate_outliers == 0:
                    st.success("No outliers found!")
                    continue

                st.error(f"Number of severe outliers: {n_severe_outliers}/{len(current_df)}")
                st.warning(f"Number of moderate outliers: {n_moderate_outliers}/{len(current_df)}")
                value = st.slider(
                    "distance to IQR",
                    min_value=float(current_df["dist_to_iqr"].min()),
                    max_value=float(current_df["dist_to_iqr"].max()),
                    step=max(
                        0.1,
                        float(
                            (current_df["dist_to_iqr"].max() - current_df["dist_to_iqr"].min())
                            / (len(current_df) / n_items_in_page)
                        ),
                    ),
                    value=float(current_df["dist_to_iqr"].max()),
                    key=f"dist_to_iqr{idx.name}",
                )

                selected_df = current_df[current_df["dist_to_iqr"] <= value][:n_items_in_page]

                cols: List = []
                for i, (row_no, row) in enumerate(selected_df.iterrows()):
                    if not cols:
                        cols = list(st.columns(n_cols))

                    with cols.pop(0):
                        image = show_image_and_draw_polygons(row)
                        st.image(image)

                        multiselect_tag(row, f"{idx.name}_summary")

                        # === Write scores and link to editor === #

                        tags_row = row.copy()
                        if row["score"] > severe_ub or row["score"] < severe_lb:
                            tags_row["outlier"] = "Severe"
                        elif row["score"] > moderate_ub or row["score"] < moderate_lb:
                            tags_row["outlier"] = "Moderate"
                        else:
                            tags_row["outlier"] = "Low"

                        if "object_class" in tags_row and not pd.isna(tags_row["object_class"]):
                            tags_row["label_class_name"] = tags_row["object_class"]
                            tags_row.drop("object_class")
                        tags_row[idx.name] = tags_row["score"]
                        build_data_tags(tags_row, idx.name)

                        if not pd.isnull(row["description"]):
                            st.write(f"Description: {row['description']}. ")
