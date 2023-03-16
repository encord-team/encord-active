from abc import abstractmethod
from typing import Optional

import streamlit as st
from loguru import logger
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator

import encord_active.app.common.state as state
from encord_active.app.common.page import Page
from encord_active.app.common.state import MetricNames
from encord_active.lib.constants import DOCS_URL
from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)
from encord_active.lib.model_predictions.reader import (
    ClassificationPredictionMatchSchema,
    LabelMatchSchema,
    PredictionMatchSchema,
)


class ModelQualityPage(Page):
    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @abstractmethod
    def sidebar_options(self):
        """
        Used to append options to the sidebar.
        """
        pass

    def sidebar_options_classifications(self):
        pass

    def check_building_object_quality(
        self,
        object_predictions_exist: bool,
        object_model_predictions: Optional[DataFrame[PredictionMatchSchema]] = None,
        object_labels: Optional[DataFrame[LabelMatchSchema]] = None,
        object_metrics: Optional[DataFrame[PerformanceMetricSchema]] = None,
        object_precisions: Optional[DataFrame[PrecisionRecallSchema]] = None,
    ) -> bool:
        if not object_predictions_exist:
            st.markdown(
                "## Missing model predictions for the objects\n"
                "This project does not have any imported predictions for the objects. "
                "Please refer to the "
                f"[Importing Model Predictions]({DOCS_URL}/sdk/importing-model-predictions) "
                "section of the documentation to learn how to import your predictions."
            )
            return False
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
            return False

        return True

    def check_building_classification_quality(
        self,
        classification_predictions_exist: bool,
        classification_labels: Optional[list] = None,
        classification_pred: Optional[list] = None,
        classification_model_predictions_matched: Optional[DataFrame[ClassificationPredictionMatchSchema]] = None,
    ) -> bool:
        if not classification_predictions_exist:
            st.markdown(
                "## Missing model predictions for the classifications\n"
                "This project does not have any imported predictions for the classifications. "
                "Please refer to the "
                f"[Importing Model Predictions]({DOCS_URL}/sdk/importing-model-predictions) "
                "section of the documentation to learn how to import your predictions."
            )
            return False
        elif not (
            (classification_labels is not None)
            and (classification_pred is not None)
            and (classification_model_predictions_matched is not None),
        ):
            logger.error(
                "If classification_predictions_exist is True, the followings should be provided: classification_labels, \
    classification_pred, classification_model_predictions_matched_filtered"
            )
            return False

        return True

    @abstractmethod
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
        classification_model_predictions_matched_filtered: Optional[
            DataFrame[ClassificationPredictionMatchSchema]
        ] = None,
    ):
        """
        If object_predictions_exist is True, the followings should be provided: object_model_predictions, \
        object_labels, object_metrics, object_precisions
        If classification_predictions_exist is True, the followings should be provided: classification_labels, \
        classification_pred, classification_model_predictions_matched_filtered
        """

        pass

    def __call__(
        self,
        model_predictions: DataFrame[PredictionMatchSchema],
        labels: DataFrame[LabelMatchSchema],
        metrics: DataFrame[PerformanceMetricSchema],
        precisions: DataFrame[PrecisionRecallSchema],
    ):
        return self.build(model_predictions, labels, metrics, precisions)

    def __repr__(self):
        return f"{type(self).__name__}()"

    @staticmethod
    def prediction_metric_in_sidebar_objects():
        """
        Note: Adding the fixed options "confidence" and "iou" works here because
        confidence is required on import and IOU is computed during prediction
        import. So the two columns are already available in the
        `st.session_state.model_predictions` data frame.
        """
        fixed_options = {"confidence": "Model Confidence", "iou": "IOU"}
        column_names = list(state.get_state().predictions.metric_datas.predictions.keys())
        state.get_state().predictions.metric_datas.selected_prediction = st.selectbox(
            "Select metric for your predictions",
            column_names + list(fixed_options.keys()),
            format_func=lambda s: fixed_options.get(s, s),
            help="The data in the main view will be sorted by the selected metric. "
            "(F) := frame scores, (P) := prediction scores.",
        )

    @staticmethod
    def prediction_metric_in_sidebar_classifications():
        fixed_options = {"confidence": "Model Confidence"}
        column_names = list(state.get_state().predictions.metric_datas_classification.predictions.keys())
        state.get_state().predictions.metric_datas_classification.selected_prediction = st.selectbox(
            "Select metric for your predictions",
            column_names + list(fixed_options.keys()),
            format_func=lambda s: fixed_options.get(s, s),
            help="The data in the main view will be sorted by the selected metric. "
            "(F) := frame scores, (P) := prediction scores.",
        )

    @staticmethod
    def metric_details_description(metric_datas: MetricNames):
        metric_name = metric_datas.selected_prediction
        if not metric_name:
            return

        metric_data = metric_datas.predictions.get(metric_name)

        if not metric_data:
            metric_data = metric_datas.labels.get(metric_name)

        if metric_data:
            st.markdown(f"### The {metric_data.name[:-4]} metric")
            st.markdown(metric_data.meta.long_description)
