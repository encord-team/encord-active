import pandas as pd
import streamlit as st

from encord_active.app.common.colors import Color
from encord_active.app.model_assertions.components import metric_view

from . import HistogramMixin, ModelAssertionsPage


class FalsePositivesPage(ModelAssertionsPage, HistogramMixin):
    title = "🌡 False Positives"

    def sidebar_options(self):
        self.prediction_metric_in_sidebar()
        self.row_col_settings_in_sidebar()

    def build(
        self,
        model_predictions: pd.DataFrame,
        labels: pd.DataFrame,
        metrics: pd.DataFrame,
        precisions: pd.DataFrame,
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
        fp_df = model_predictions[model_predictions["tps"] == 0.0].dropna(subset=[metric_name])
        if fp_df.shape[0] == 0:
            st.write("No false positives")
        else:
            histogram = self.get_histogram(fp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            metric_view(fp_df, box_color=color)
