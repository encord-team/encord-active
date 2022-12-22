import pandas as pd
import streamlit as st

from encord_active.app.model_quality.components import metric_view
from encord_active.lib.common.colors import Color
from encord_active.lib.metrics.statistical_utils import get_histogram

from . import ModelQualityPage


class TruePositivesPage(ModelQualityPage):
    title = "✅ True Positives"

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
        metric_name = st.session_state.predictions_metric
        tp_df = model_predictions[model_predictions["tps"] == 1.0].dropna(subset=[metric_name])
        if tp_df.shape[0] == 0:
            st.write("No true positives")
        else:
            histogram = get_histogram(tp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            metric_view(tp_df, box_color=color)
