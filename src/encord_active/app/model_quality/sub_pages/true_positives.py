from typing import Optional, cast

import streamlit as st
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator

from encord_active.app.common.components.prediction_grid import (
    prediction_grid,
    prediction_grid_classifications,
)
from encord_active.app.common.state import get_state
from encord_active.lib.charts.histogram import get_histogram
from encord_active.lib.common.colors import Color
from encord_active.lib.metrics.utils import MetricScope
from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)
from encord_active.lib.model_predictions.reader import (
    ClassificationPredictionMatchSchemaWithClassNames,
    LabelMatchSchema,
    PredictionMatchSchema,
)

from . import ModelQualityPage


class TruePositivesPage(ModelQualityPage):
    title = "✅ True Positives"

    def sidebar_options(self):
        self.prediction_metric_in_sidebar_objects()
        self.display_settings(MetricScope.MODEL_QUALITY)

    def sidebar_options_classifications(self):
        self.prediction_metric_in_sidebar_classifications()

    def _build_objects(
        self,
        object_model_predictions: DataFrame[PredictionMatchSchema],
    ):
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
            self.metric_details_description(get_state().predictions.metric_datas)

        metric_name = get_state().predictions.metric_datas.selected_prediction
        if not metric_name:
            st.error("No prediction metric selected")
            return

        tp_df = object_model_predictions[
            object_model_predictions[PredictionMatchSchema.is_true_positive] == 1.0
        ].dropna(subset=[metric_name])
        if tp_df.shape[0] == 0:
            st.write("No true positives")
        else:
            histogram = get_histogram(tp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid(get_state().project_paths.data, model_predictions=tp_df, box_color=color)

    def _build_classifications(
        self,
        classification_model_predictions_matched: DataFrame[ClassificationPredictionMatchSchemaWithClassNames],
    ):
        with st.expander("Details"):
            st.markdown(
                """### The view
These are the predictions where the model correctly predicts the true class.
                    """,
                unsafe_allow_html=True,
            )
            self.metric_details_description(get_state().predictions.metric_datas_classification)

        metric_name = get_state().predictions.metric_datas_classification.selected_prediction
        if not metric_name:
            st.error("No prediction metric selected")
            return

        tp_df = classification_model_predictions_matched[
            classification_model_predictions_matched[ClassificationPredictionMatchSchemaWithClassNames.is_true_positive]
            == 1.0
        ].dropna(subset=[metric_name])

        if tp_df.shape[0] == 0:
            st.write("No true positives")
        else:
            histogram = get_histogram(tp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid_classifications(get_state().project_paths.data, model_predictions=tp_df)

    def build(
        self,
        object_predictions_exist: bool,
        classification_predictions_exist: bool,
        object_tab: DeltaGenerator,
        classification_tab: DeltaGenerator,
        object_model_predictions: Optional[DataFrame[PredictionMatchSchema]] = None,
        object_labels: Optional[DataFrame[LabelMatchSchema]] = None,
        object_metrics: Optional[DataFrame[PerformanceMetricSchema]] = None,
        object_precisions: Optional[DataFrame[PrecisionRecallSchema]] = None,
        classification_labels: Optional[list] = None,
        classification_pred: Optional[list] = None,
        classification_model_predictions_matched: Optional[
            DataFrame[ClassificationPredictionMatchSchemaWithClassNames]
        ] = None,
    ):
        with object_tab:
            if self.check_building_object_quality(
                object_predictions_exist, object_model_predictions, object_labels, object_metrics, object_precisions
            ):
                self._build_objects(cast(DataFrame[PredictionMatchSchema], object_model_predictions))

        with classification_tab:
            if self.check_building_classification_quality(
                classification_predictions_exist,
                classification_labels,
                classification_pred,
                classification_model_predictions_matched,
            ):
                self._build_classifications(
                    cast(
                        DataFrame[ClassificationPredictionMatchSchemaWithClassNames],
                        classification_model_predictions_matched,
                    )
                )
