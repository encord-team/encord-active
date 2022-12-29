import streamlit as st
from pandera.typing import DataFrame

from encord_active.app.common.components.prediction_grid import prediction_grid
from encord_active.lib.charts.histogram import get_histogram
from encord_active.lib.common.colors import Color
from encord_active.lib.model_predictions.reader import (
    LabelMatchSchema,
    PredictionMatchSchema,
)
from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)

from . import ModelQualityPage


class FalsePositivesPage(ModelQualityPage):
    title = "🌡 False Positives"

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
        color = Color.RED
        with st.expander("Details"):
            st.markdown(
                f"""### The view
These are the predictions for which either of the following is true
1. The IOU between the prediction and the best matching label was too low
2. There was another prediction with higher model confidence which matched the label already
3. The predicted class didn't match

---

**Color**:
The <span style="border: solid 3px {color.value}; padding: 2px 3px 3px 3px; border-radius: 4px; color: {color.value}; font-weight: bold;">{color.name.lower()}</span> boxes marks the false positive predictions.
The remaining colors correspond to the dataset labels with the colors you are used to from the label editor.
""",
                unsafe_allow_html=True,
            )
            self.metric_details_description()

        metric_name = st.session_state.predictions_metric
        fp_df = model_predictions[model_predictions[PredictionMatchSchema.is_true_positive] == 0.0].dropna(
            subset=[metric_name]
        )
        if fp_df.shape[0] == 0:
            st.write("No false positives")
        else:
            histogram = get_histogram(fp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid(st.session_state.data_dir, model_predictions=fp_df, box_color=color)
