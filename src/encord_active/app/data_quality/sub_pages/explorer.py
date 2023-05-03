from typing import List, Optional

import streamlit as st
from encord_active_components.components.explorer import explorer
from natsort import natsorted
from pandera.typing import DataFrame

from encord_active.app.actions_page.export_filter import render_filter
from encord_active.app.common.components.annotator_statistics import (
    render_annotator_properties,
)
from encord_active.app.common.components.label_statistics import (
    render_dataset_properties,
)
from encord_active.app.common.page import Page
from encord_active.app.common.state import get_state
from encord_active.lib.metrics.utils import (
    MetricData,
    MetricSchema,
    MetricScope,
    filter_none_empty_metrics,
    load_metric_dataframe,
)
from encord_active.server.settings import settings


class ExplorerPage(Page):
    title = "🔎 Explorer"

    def sidebar_options(self, available_metrics: List[MetricData], *_) -> Optional[DataFrame[MetricSchema]]:
        self.available_metrics = available_metrics
        self.display_settings()

        selected_metric = get_state().filtering_state.sort_by_metric
        if not selected_metric:
            return None

        df = load_metric_dataframe(selected_metric)

        selected_classes = get_state().filtering_state.selected_classes
        is_class_selected = (
            df.shape[0] * [True] if not selected_classes else df[MetricSchema.object_class].isin(selected_classes)
        )
        df = df[is_class_selected]

        selected_annotators = get_state().filtering_state.selected_annotators
        annotator_selected = (
            df.shape[0] * [True] if not selected_annotators else df[MetricSchema.annotator].isin(selected_annotators)
        )

        df = df[annotator_selected].pipe(DataFrame[MetricSchema])

        fmm = get_state().filtering_state.merged_metrics
        return df.set_index("identifier").loc[fmm.index[fmm.index.isin(df.identifier)]].reset_index()

    def build(self, selected_df: DataFrame[MetricSchema], metric_scope: MetricScope):
        selected_metric = get_state().filtering_state.sort_by_metric
        if not selected_metric:
            return

        with st.expander("Dataset Properties", expanded=True):
            render_dataset_properties(selected_df)
        with st.expander("Annotator Statistics", expanded=False):
            render_annotator_properties(selected_df)

        explorer(
            project_name=get_state().project_paths.project_dir.name,
            items=[id for id in get_state().filtering_state.merged_metrics.index.values],
            scope=metric_scope.value,
            api_url=settings.API_URL,
        )

    def render_view_options(self):
        non_empty_metrics = list(filter(filter_none_empty_metrics, self.available_metrics))
        metric_data_options = natsorted(non_empty_metrics, key=lambda i: i.name)

        if not metric_data_options:
            return

        get_state().filtering_state.sort_by_metric = metric_data_options[0]
        render_filter()
