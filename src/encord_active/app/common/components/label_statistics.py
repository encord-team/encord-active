import pandas as pd
import plotly.express as px
import streamlit as st
from natsort import natsorted
from pandera.typing import DataFrame

from encord_active.lib.metrics.utils import MetricSchema


def render_dataset_properties(current_df: DataFrame[MetricSchema]):
    cls_set = natsorted(current_df[MetricSchema.object_class].dropna().unique().tolist())
    if len(cls_set) == 0:
        return

    dataset_columns = st.columns(3)
    dataset_columns[0].metric("Number of labels", current_df.shape[0])
    dataset_columns[1].metric("Number of classes", len(cls_set))
    dataset_columns[2].metric("Number of images", get_unique_data_units_size(current_df))

    if len(cls_set) > 1:
        classes = {}
        for cls in cls_set:
            classes[cls] = (current_df[MetricSchema.object_class] == cls).sum()

        source = pd.DataFrame({"class": list(classes.keys()), "count": list(classes.values())})

        fig = px.bar(source, x="class", y="count")
        fig.update_layout(title_text="Distribution of the classes", title_x=0.5, title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)


def get_unique_data_units_size(current_df: pd.DataFrame):
    data_units = set()
    identifiers = current_df["identifier"]
    for identifier in identifiers:
        key_components = identifier.split("_")
        data_units.add(key_components[0] + "_" + key_components[1])

    return len(data_units)
