from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from pandera.typing import DataFrame
from streamlit_plotly_events import plotly_events

from encord_active.app.common.state_hooks import UseState
from encord_active.lib.embeddings.utils import (
    Embedding2DSchema,
    PointSchema2D,
    PointSelectionSchema,
)


def get_selected_rows(
    embeddings_2d: DataFrame[Embedding2DSchema], selected_points: list[dict]
) -> DataFrame[Embedding2DSchema]:
    selection_raw = DataFrame[PointSelectionSchema](pd.DataFrame(selected_points))
    selected_rows = embeddings_2d.copy().merge(
        selection_raw[[PointSchema2D.x, PointSchema2D.y]],
        on=[PointSchema2D.x, PointSchema2D.y],
        how="inner",
    )
    return DataFrame[Embedding2DSchema](selected_rows)


def render_plotly_events(embedding_2d: DataFrame[Embedding2DSchema]) -> Optional[DataFrame[Embedding2DSchema]]:
    should_select = UseState(True)
    selection = UseState[Optional[List[dict]]](None)

    fig = px.scatter(
        embedding_2d,
        x=Embedding2DSchema.x,
        y=Embedding2DSchema.y,
        color=Embedding2DSchema.label,
        color_discrete_map={"True prediction": "#5658dd", "False prediction": "#ff1a1a"},
        title="2D embedding plot",
        template="plotly",
    )

    new_selection = plotly_events(fig, click_event=False, select_event=True)

    if new_selection != selection.value:
        should_select.set(True)
        selection.set(new_selection)

    if st.button("Reset selection"):
        should_select.set(False)

    if should_select.value and len(new_selection) > 0:
        return get_selected_rows(embedding_2d, new_selection)
    else:
        return None
