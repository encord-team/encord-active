from typing import List

from encord_active.app.common.components.metric_summary import (
    render_data_quality_dashboard,
    render_label_quality_dashboard,
)
from encord_active.app.common.page import Page
from encord_active.lib.metrics.utils import MetricData, MetricScope


class SummaryPage(Page):
    title = "📑 Summary"
    _severe_outlier_color = "tomato"
    _moderate_outlier_color = "orange"
    _summary_item_background_color = "#fbfbfb"

    def sidebar_options(self, *_):
        self.display_settings()

    def build(self, metrics: List[MetricData], metric_scope: MetricScope):
        if metric_scope == MetricScope.DATA_QUALITY:
            render_data_quality_dashboard(
                metrics, self._severe_outlier_color, self._moderate_outlier_color, self._summary_item_background_color
            )
        elif metric_scope == MetricScope.LABEL_QUALITY:
            render_label_quality_dashboard(
                metrics, self._severe_outlier_color, self._moderate_outlier_color, self._summary_item_background_color
            )
