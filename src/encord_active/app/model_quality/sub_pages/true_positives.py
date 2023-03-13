from typing import Optional

import streamlit as st
from loguru import logger
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator

from encord_active.app.common.components.prediction_grid import prediction_grid
from encord_active.app.common.state import get_state
from encord_active.lib.charts.histogram import get_histogram
from encord_active.lib.common.colors import Color
from encord_active.lib.constants import DOCS_URL
from encord_active.lib.metrics.utils import MetricScope
from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)
from encord_active.lib.model_predictions.reader import (
    ClassificationPredictionMatchSchema,
    LabelMatchSchema,
    PredictionMatchSchema,
)

from . import ModelQualityPage


class TruePositivesPage(ModelQualityPage):
    title = "✅ True Positives"

    def sidebar_options(self):
        self.prediction_metric_in_sidebar_objects()
        self.display_settings(MetricScope.MODEL_QUALITY)

    def sidebar_options_classifications(slef):
        pass

    def _build_objects(
        self,
        object_model_predictions: Optional[DataFrame[PredictionMatchSchema]],
        object_labels: Optional[DataFrame[LabelMatchSchema]],
        object_metrics: Optional[DataFrame[PerformanceMetricSchema]],
        object_precisions: Optional[DataFrame[PrecisionRecallSchema]],
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
            self.metric_details_description()

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
        classification_labels: Optional[list],
        classification_pred: Optional[list],
        classification_model_predictions_matched: Optional[DataFrame[ClassificationPredictionMatchSchema]],
    ):
        pass

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
        classification_model_predictions_matched: Optional[DataFrame[ClassificationPredictionMatchSchema]] = None,
    ):
        """
                If object_prediction_exist is True, the followings should be provided: object_model_predictions, \
                object_labels, object_metrics, object_precisions
                If classification_predictions_exist is True, the followings should be provided: classification_labels, \
                classification_pred, classification_model_predictions_matched_filtered
                """

        with object_tab:
            if not object_predictions_exist:
                st.markdown(
                    "## Missing model predictions for the classifications\n"
                    "This project does not have any imported predictions for the classifications. "
                    "Please refer to the "
                    f"[Importing Model Predictions]({DOCS_URL}/sdk/importing-model-predictions) "
                    "section of the documentation to learn how to import your predictions."
                )
            elif not (
                (object_model_predictions is not None)
                and (object_labels is not None)
                and (object_metrics is not None)
                and (object_precisions is not None)
            ):
                logger.error(
                    "If object_prediction_exist is True, the followings should be provided: object_model_predictions, \
        object_labels, object_metrics, object_precisions"
                )
            else:
                self._build_objects(object_model_predictions, object_labels, object_metrics, object_precisions)

        with classification_tab:
            if not classification_predictions_exist:
                st.markdown(
                    "## Missing model predictions for the classifications\n"
                    "This project does not have any imported predictions for the classifications. "
                    "Please refer to the "
                    f"[Importing Model Predictions]({DOCS_URL}/sdk/importing-model-predictions) "
                    "section of the documentation to learn how to import your predictions."
                )
            elif not (
                (classification_labels is not None)
                and (classification_pred is not None)
                and (classification_model_predictions_matched is not None),
            ):
                logger.error(
                    "If classification_predictions_exist is True, the followings should be provided: classification_labels, \
        classification_pred, classification_model_predictions_matched_filtered"
                )
            else:
                self._build_classifications(
                    classification_labels, classification_pred, classification_model_predictions_matched
                )
