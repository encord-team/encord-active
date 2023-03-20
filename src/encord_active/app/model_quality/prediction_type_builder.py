from abc import abstractmethod
from enum import Enum

import streamlit as st

from encord_active.app.common.page import Page
from encord_active.app.common.state import MetricNames


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
    def _load_data(self, page_mode: ModelQualityPage):
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

    def build(self, page_mode: ModelQualityPage):
        self._load_data(page_mode)

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
