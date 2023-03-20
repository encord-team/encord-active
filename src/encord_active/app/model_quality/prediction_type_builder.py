from enum import Enum

import streamlit as st

from encord_active.app.common.state import MetricNames


class ModelQualityPage(str, Enum):
    METRICS = "metrics"
    PERFORMANCE_BY_METRIC = "performance by metric"
    TRUE_POSITIVES = "true positives"
    FALSE_POSITIVES = "false positives"
    FALSE_NEGATIVES = "false negatives"


class PredictionTypeBuilder:
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

    def render(self, page_mode: ModelQualityPage):
        pass

    def common_settings(self):
        pass

    def is_available(self) -> bool:
        pass
