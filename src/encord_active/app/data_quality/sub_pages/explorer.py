import json
import re
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple
from urllib import parse

import pandas as pd
import streamlit as st
from encord_active_components.components.explorer import (
    GroupedTags,
    Output,
    OutputAction,
    explorer,
)
from natsort import natsorted
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator
from streamlit.elements.image import image_to_url

from encord_active.app.actions_page.export_filter import render_filter
from encord_active.app.common.components import build_data_tags, divider
from encord_active.app.common.components.annotator_statistics import (
    render_annotator_properties,
)
from encord_active.app.common.components.interative_plots import render_plotly_events
from encord_active.app.common.components.label_statistics import (
    render_dataset_properties,
)
from encord_active.app.common.components.paginator import paginate_df
from encord_active.app.common.components.similarities import show_similarities
from encord_active.app.common.components.tags.bulk_tagging_form import (
    BulkLevel,
    action_bulk_tags,
    bulk_tagging_form,
)
from encord_active.app.common.components.tags.individual_tagging import multiselect_tag
from encord_active.app.common.page import Page
from encord_active.app.common.state import get_state
from encord_active.app.common.state_hooks import UseState
from encord_active.lib.charts.histogram import get_histogram
from encord_active.lib.common.image_utils import (
    ObjectDrawingConfigurations,
    get_geometries,
    show_image_and_draw_polygons,
)
from encord_active.lib.db.helpers.tags import to_grouped_tags
from encord_active.lib.db.merged_metrics import MANDATORY_COLUMNS
from encord_active.lib.db.tags import Tag, Tags, TagScope
from encord_active.lib.embeddings.dimensionality_reduction import get_2d_embedding_data
from encord_active.lib.embeddings.utils import Embedding2DSchema, SimilaritiesFinder
from encord_active.lib.metrics.metric import EmbeddingType
from encord_active.lib.metrics.utils import (
    IdentifierSchema,
    MetricData,
    MetricSchema,
    MetricScope,
    filter_none_empty_metrics,
    get_embedding_type,
    load_metric_dataframe,
)


class ExplorerPage(Page):
    title = "🔎 Explorer"

    def sidebar_options(
        self, available_metrics: List[MetricData], metric_scope: MetricScope
    ) -> Optional[DataFrame[MetricSchema]]:
        self.available_metrics = available_metrics
        self.display_settings(metric_scope == MetricScope.DATA_QUALITY)

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

        embedding_type = get_embedding_type(selected_metric.meta.annotation_type)

        with_all_metrics = (
            selected_df[["identifier"]].join(get_state().merged_metrics, on="identifier", how="left").dropna(axis=1)
        )

        output_state = UseState[Optional[Output]](None)
        output = explorer(
            project_name=get_state().project_paths.project_dir.name,
            items=[id for id in with_all_metrics["identifier"].values],
            all_tags=to_grouped_tags(Tags().all()),
            scope=metric_scope.value,
            embeddings_type=embedding_type.value,
        )
        if output and output != output_state:
            output_state.set(output)
            action, payload = output
            print(output)

    def render_view_options(self):
        non_empty_metrics = list(filter(filter_none_empty_metrics, self.available_metrics))
        metric_data_options = natsorted(non_empty_metrics, key=lambda i: i.name)

        if not metric_data_options:
            return

        get_state().filtering_state.sort_by_metric = metric_data_options[0]
        render_filter()
