import pandas as pd
import streamlit as st

import encord_active.app.common.state as state
from encord_active.app.common.colors import Color
from encord_active.app.model_assertions.components import false_negative_view

from . import HistogramMixin, ModelAssertionsPage


class FalseNegativesPage(ModelAssertionsPage, HistogramMixin):
    title = "🔍 False Negatives"

    def sidebar_options(self):
        st.selectbox(
            "Select metric for your labels",
            st.session_state.label_metric_names,
            key=state.PREDICTIONS_LABEL_METRIC,
            help="The data in the main view will be sorted by the selected metric. "
            "(F) := frame scores, (O) := object scores.",
        )
        self.row_col_settings_in_sidebar()

    def build(
        self,
        model_predictions: pd.DataFrame,
        labels: pd.DataFrame,
        metrics: pd.DataFrame,
        precisions: pd.DataFrame,
    ):
        st.markdown(f"# {self.title}")
        st.header("False Negatives")
        metric_name = st.session_state[state.PREDICTIONS_LABEL_METRIC]
        with st.expander("Details"):
            color = Color.PURPLE
            st.markdown(
                f"""### The view
These are the labels that were not matched with any predictions.

---
**Color**:
The <span style="border: solid 3px {color.value}; padding: 2px 3px 3px 3px; border-radius: 4px; color: {color.value}; font-weight: bold;">{color.name.lower()}</span> boxes mark the false negatives.
That is, the labels that were not matched to any predictions.
The remaining objects are predictions, where colors correspond to their predicted class (identical colors to labels objects in the editor).
""",
                unsafe_allow_html=True,
            )
            self.metric_details_description(metric_name)
        fns_df = labels[labels["fns"]].dropna(subset=[metric_name])
        if fns_df.shape[0] == 0:
            st.write("No false negatives")
        else:
            histogram = self.get_histogram(fns_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            false_negative_view(fns_df, model_predictions, color=color)
