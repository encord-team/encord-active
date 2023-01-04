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


class TruePositivesPage(ModelQualityPage):
    title = "✅ True Positives"

    def sidebar_options(self):
        self.prediction_metric_in_sidebar()
        self.row_col_settings_in_sidebar()

    def build(
        self,
        model_predictions: DataFrame[PredictionMatchSchema],
        labels: DataFrame[LabelMatchSchema],
        metrics: DataFrame[PerformanceMetricSchema],
        precisions: DataFrame[PrecisionRecallSchema],
    ):
        st.markdown(f"# {self.title}")

        with st.expander("Details"):
            color = Color.PURPLE
            st.markdown(
                f"""### The view
These are the predictions for which the IOU was sufficiently high and the confidence score was
the highest amongst predictions that overlap with the label.

---

**Color**:
The <span style="border: solid 3px {color.value}; padding: 2px 3px 3px 3px; border-radius: 4px; color: {color.value}; font-weight: bold;">{color.name.lower()}</span> boxes marks the true positive predictions.
The remaining colors correspond to the dataset labels with the colors you are used to from the label editor.
            """,
                unsafe_allow_html=True,
            )
            self.metric_details_description()

        metric_name = get_state().predictions.metric_datas.selected_predicion
        if not metric_name:
            st.error("No prediction metric selected")
            return

        tp_df = model_predictions[model_predictions[PredictionMatchSchema.is_true_positive] == 1.0].dropna(
            subset=[metric_name]
        )
        if tp_df.shape[0] == 0:
            st.write("No true positives")
        else:
            histogram = get_histogram(tp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid(get_state().project_paths.data, model_predictions=tp_df, box_color=color)
