from abc import abstractmethod
from enum import Enum
from typing import List, Union

import streamlit as st
from pandera.typing import DataFrame

import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.page import Page
from encord_active.app.common.state import MetricNames, get_state
from encord_active.app.common.state_hooks import use_memo
from encord_active.lib.metrics.utils import MetricData
from encord_active.lib.model_predictions.reader import (
    ClassificationLabelSchema,
    ClassificationPredictionSchema,
    LabelSchema,
    PredictionSchema,
)
from encord_active.lib.model_predictions.writer import MainPredictionType


class ModelQualityPage(str, Enum):
    METRICS = "metrics"
    PERFORMANCE_BY_METRIC = "performance by metric"
    TRUE_POSITIVES = "true positives"
    FALSE_POSITIVES = "false positives"
    FALSE_NEGATIVES = "false negatives"


class PredictionTypeBuilder(Page):
    name: str

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

    @abstractmethod
    def _load_data(self, page_mode: ModelQualityPage) -> bool:
        pass

    @abstractmethod
    def _render_metrics(self):
        pass

    @abstractmethod
    def _render_performance_by_metric(self):
        pass

    @abstractmethod
    def _render_true_positives(self):
        pass

    @abstractmethod
    def _render_false_positives(self):
        pass

    @abstractmethod
    def _render_false_negatives(self):
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    def read_prediction_files(
        self, prediction_type: MainPredictionType
    ) -> (
        List[MetricData],
        List[MetricData],
        Union[DataFrame[PredictionSchema], DataFrame[ClassificationPredictionSchema], None],
        Union[DataFrame[LabelSchema], DataFrame[ClassificationLabelSchema], None],
    ):
        metrics_dir = get_state().project_paths.metrics
        predictions_dir = get_state().project_paths.predictions / prediction_type.value

        predictions_metric_datas = use_memo(lambda: reader.get_prediction_metric_data(predictions_dir, metrics_dir))

        label_metric_datas = use_memo(lambda: reader.get_label_metric_data(metrics_dir))
        model_predictions = use_memo(
            lambda: reader.get_model_predictions(predictions_dir, predictions_metric_datas, prediction_type)
        )
        labels = use_memo(lambda: reader.get_labels(predictions_dir, label_metric_datas, prediction_type))

        return predictions_metric_datas, label_metric_datas, model_predictions, labels

    def build(self, page_mode: ModelQualityPage):
        if self._load_data(page_mode):
            if page_mode == ModelQualityPage.METRICS:
                self._render_metrics()
            elif page_mode == ModelQualityPage.PERFORMANCE_BY_METRIC:
                self._render_performance_by_metric()
            elif page_mode == ModelQualityPage.TRUE_POSITIVES:
                self._render_true_positives()
            elif page_mode == ModelQualityPage.FALSE_POSITIVES:
                self._render_false_positives()
            elif page_mode == ModelQualityPage.FALSE_NEGATIVES:
                self._render_false_negatives()
