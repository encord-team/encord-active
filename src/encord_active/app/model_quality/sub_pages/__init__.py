from abc import abstractmethod
from typing import List, Optional

import streamlit as st
from pandera.typing import DataFrame

import encord_active.app.common.state as state
from encord_active.app.common.page import Page
from encord_active.lib.metrics.utils import MetricData
from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)
from encord_active.lib.model_predictions.reader import (
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

    @abstractmethod
    def build(
        self,
        model_predictions: DataFrame[PredictionMatchSchema],
        labels: DataFrame[LabelMatchSchema],
        metrics: DataFrame[PerformanceMetricSchema],
        precisions: DataFrame[PrecisionRecallSchema],
    ):
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
    def prediction_metric_in_sidebar():
        """
        Note: Adding the fixed options "confidence" and "iou" works here because
        confidence is required on import and IOU is computed during prediction
        import. So the two columns are already available in the
        `st.session_state.model_predictions` data frame.
        """
        fixed_options = {"confidence": "Model Confidence", "iou": "IOU"}
        column_names = list(state.get_state().predictions.metric_datas.predictions.keys())
        state.get_state().predictions.metric_datas.selected_predicion = st.selectbox(
            "Select metric for your predictions",
            column_names + list(fixed_options.keys()),
            format_func=lambda s: fixed_options.get(s, s),
            help="The data in the main view will be sorted by the selected metric. "
            "(F) := frame scores, (P) := prediction scores.",
        )

    @staticmethod
    def metric_details_description():
        metric_name = state.get_state().predictions.metric_datas.selected_predicion
        if not metric_name:
            return

        metric_data = state.get_state().predictions.metric_datas.predictions.get(metric_name)

        if not metric_data:
            metric_data = state.get_state().predictions.metric_datas.labels.get(metric_name)

        if metric_data:
            st.markdown(f"### The {metric_data.name[:-4]} metric")
            st.markdown(metric_data.meta.long_description)
