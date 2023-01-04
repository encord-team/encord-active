import streamlit as st
from pandera.typing import DataFrame

from encord_active.app.common.components.prediction_grid import prediction_grid
from encord_active.app.common.state import get_state
from encord_active.lib.charts.histogram import get_histogram
from encord_active.lib.common.colors import Color
from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)
from encord_active.lib.model_predictions.reader import (
    LabelMatchSchema,
    PredictionMatchSchema,
)

from . import ModelQualityPage


class FalseNegativesPage(ModelQualityPage):
    title = "🔍 False Negatives"

    def sidebar_options(self):
        metric_columns = list(get_state().predictions.metric_datas.labels.keys())
        get_state().predictions.metric_datas.selected_label = st.selectbox(
            "Select metric for your labels",
            metric_columns,
            help="The data in the main view will be sorted by the selected metric. "
            "(F) := frame scores, (O) := object scores.",
        )
        self.row_col_settings_in_sidebar()

    def build(
        self,
        model_predictions: DataFrame[PredictionMatchSchema],
        labels: DataFrame[LabelMatchSchema],
        metrics: DataFrame[PerformanceMetricSchema],
        precisions: DataFrame[PrecisionRecallSchema],
    ):
        st.markdown(f"# {self.title}")
        st.header("False Negatives")
        metric_name = get_state().predictions.metric_datas.selected_label
        if not metric_name:
            st.error("Prediction label not selected")
            return

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
            self.metric_details_description()
        fns_df = labels[labels[LabelMatchSchema.is_false_negative]].dropna(subset=[metric_name])
        if fns_df.shape[0] == 0:
            st.write("No false negatives")
        else:
            histogram = get_histogram(fns_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid(
                get_state().project_paths.data, labels=fns_df, model_predictions=model_predictions, box_color=color
            )
