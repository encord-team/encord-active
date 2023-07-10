import pandas as pd
import plotly.express as px
import streamlit as st
from pandera.typing import DataFrame

from encord_active.lib.metrics.utils import (
    AnnotatorInfo,
    MetricSchema,
    get_annotator_level_info,
)


def render_annotator_properties(df: DataFrame[MetricSchema]):
    annotators = get_annotator_level_info(df)
    if len(annotators) == 0:
        st.warning("Found no annotators")
        return

    annotators_df = pd.DataFrame(list(annotators.values()))
    total_annotations = annotators_df["total_annotations"].sum()
    if not total_annotations:
        st.warning("No annotations to show")
        return

    left_col, right_col = st.columns([2, 2])
    # 1. Pie Chart
    left_col.markdown(
        "<h5 style='text-align: center; color: black;'>Distribution of annotations</h1>", unsafe_allow_html=True
    )

    fig = px.pie(annotators_df, values="total_annotations", names="name", hover_data=["mean_score"])

    left_col.plotly_chart(fig, use_container_width=True)

    # 2. Table View
    right_col.markdown(
        "<h5 style='text-align: center; color: black;'>Detailed annotator statistics</h1>", unsafe_allow_html=True
    )

    total_mean_score = (annotators_df["mean_score"] * annotators_df["total_annotations"]).sum() / total_annotations

    info = AnnotatorInfo(name="all", total_annotations=total_annotations, mean_score=total_mean_score)
    annotators_df = pd.concat([annotators_df, pd.Series(info).to_frame().T], ignore_index=True)

    annotators_df["deviation"] = (annotators_df["mean_score"] - total_mean_score) / (total_mean_score or 1) * 100

    right_col.dataframe(annotators_df.style.pipe(make_pretty), use_container_width=True)


def _format_deviation(val):
    return f"{val:.1f}%"


def _format_score(val):
    return f"{val:.3f}"


def _color_red_or_green(val):
    color = "red" if val < 0 else "green"
    return f"color: {color}"


def make_pretty(styler):
    styler.format(_format_deviation, subset=["deviation"])
    styler.format(_format_score, subset=["mean_score"])
    styler.applymap(_color_red_or_green, subset=["deviation"])
    return styler
