import re
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from natsort import natsorted
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator
from streamlit_plotly_events import plotly_events

from encord_active.app.common.components import build_data_tags, divider
from encord_active.app.common.components.annotator_statistics import (
    render_annotator_properties,
)
from encord_active.app.common.components.label_statistics import (
    render_dataset_properties,
)
from encord_active.app.common.components.paginator import render_pagination
from encord_active.app.common.components.similarities import show_similarities
from encord_active.app.common.components.slicer import render_df_slicer
from encord_active.app.common.components.tags.bulk_tagging_form import (
    BulkLevel,
    action_bulk_tags,
    bulk_tagging_form,
)
from encord_active.app.common.components.tags.individual_tagging import multiselect_tag
from encord_active.app.common.components.tags.tag_creator import tag_creator
from encord_active.app.common.page import Page
from encord_active.app.common.state import get_state
from encord_active.app.common.state_hooks import use_state
from encord_active.app.label_onboarding.label_onboarding import label_onboarding_page
from encord_active.lib.charts.histogram import get_histogram
from encord_active.lib.common.image_utils import (
    load_or_fill_image,
    show_image_and_draw_polygons,
)
from encord_active.lib.embeddings.dimensionality_reduction import get_2d_embedding_data
from encord_active.lib.embeddings.utils import (
    Embedding2DSchema,
    PointSchema2D,
    PointSelectionSchema,
    SimilaritiesFinder,
)
from encord_active.lib.metrics.metric import EmbeddingType
from encord_active.lib.metrics.utils import (
    MetricData,
    MetricSchema,
    MetricScope,
    get_annotator_level_info,
    get_embedding_type,
    load_metric_dataframe,
)


class ExplorerPage(Page):
    title = "🔎 Explorer"

    def sidebar_options(self, available_metrics: List[MetricData]):
        tag_creator()

        if not available_metrics:
            return label_onboarding_page()

        non_empty_metrics = [
            metric for metric in available_metrics if not load_metric_dataframe(metric, normalize=False).empty
        ]
        sorted_metrics = natsorted(non_empty_metrics, key=lambda i: i.name)

        metric_names = list(map(lambda i: i.name, sorted_metrics))

        col1, col2, col3 = st.columns(3)
        selected_metric_name = col1.selectbox(
            "Select a metric to order your data by",
            metric_names,
            help="The data in the main view will be sorted by the selected metric. ",
        )

        if not selected_metric_name:
            return

        metric_idx = metric_names.index(selected_metric_name)
        selected_metric = sorted_metrics[metric_idx]
        get_state().selected_metric = selected_metric

        df = load_metric_dataframe(selected_metric)
        if df.shape[0] <= 0:
            return

        class_set = natsorted(df[MetricSchema.object_class].dropna().unique().tolist())
        with col2:
            selected_classes = None
            if len(class_set) > 0:
                selected_classes = st.multiselect("Filter by class", class_set)

        is_class_selected = (
            df.shape[0] * [True] if not selected_classes else df[MetricSchema.object_class].isin(selected_classes)
        )
        df_class_selected: DataFrame[MetricSchema] = df[is_class_selected]

        annotators = get_annotator_level_info(df_class_selected)
        annotator_set = natsorted(annotators.keys())
        with col3:
            selected_annotators = None
            if len(annotator_set) > 0:
                selected_annotators = st.multiselect("Filter by annotator", annotator_set)

        annotator_selected = (
            df_class_selected.shape[0] * [True]
            if not selected_annotators
            else df_class_selected[MetricSchema.annotator].isin(selected_annotators)
        )

        self.row_col_settings_in_sidebar()
        # For now go the easy route and just filter the dataframe here
        return df_class_selected[annotator_selected]

    def build(self, selected_df: DataFrame[MetricSchema], metric_scope: MetricScope):
        selected_metric = get_state().selected_metric
        if not selected_metric:
            return

        st.markdown(f"# {self.title}")
        st.markdown(f"## {selected_metric.meta.title}")
        st.markdown(selected_metric.meta.long_description)

        if selected_df.empty:
            return

        with st.expander("Dataset Properties", expanded=True):
            render_dataset_properties(selected_df)
        with st.expander("Annotator Statistics", expanded=False):
            render_annotator_properties(selected_df)

        fill_data_quality_window(selected_df, metric_scope, selected_metric)


def get_selected_rows(
    embeddings_2d: DataFrame[Embedding2DSchema], selected_points: list[dict]
) -> DataFrame[Embedding2DSchema]:
    selection_raw = DataFrame[PointSelectionSchema](pd.DataFrame(selected_points))
    selected_rows = embeddings_2d.copy().merge(
        selection_raw[[PointSchema2D.x, PointSchema2D.y]],
        on=[PointSchema2D.x, PointSchema2D.y],
        how="inner",
    )
    return DataFrame[Embedding2DSchema](selected_rows)


def render_plotly_events(embedding_2d: DataFrame[Embedding2DSchema]) -> Optional[DataFrame[Embedding2DSchema]]:
    get_should_select, set_should_select = use_state(True)
    get_selection, set_selection = use_state(None)
    fig = px.scatter(
        embedding_2d,
        x=Embedding2DSchema.x,
        y=Embedding2DSchema.y,
        color=Embedding2DSchema.label,
        title="2D embedding plot",
    )

    new_selection = plotly_events(fig, click_event=False, select_event=True)

    if new_selection != get_selection():
        set_should_select(True)
        set_selection(new_selection)

    if st.button("Reset selection"):
        set_should_select(False)

    if get_should_select() and len(new_selection) > 0:
        return get_selected_rows(embedding_2d, new_selection)
    else:
        return None


def fill_data_quality_window(
    current_df: DataFrame[MetricSchema], metric_scope: MetricScope, selected_metric: MetricData
):
    meta = selected_metric.meta
    embedding_type = get_embedding_type(meta.title, meta.annotation_type)
    embeddings_dir = get_state().project_paths.embeddings
    embedding_information = SimilaritiesFinder(embedding_type, embeddings_dir)

    if (embedding_information.type == EmbeddingType.CLASSIFICATION) and len(embedding_information.collections) == 0:
        st.write("Image-level embedding file is not available for this project.")
        return
    if (embedding_information.type == EmbeddingType.OBJECT) and len(embedding_information.collections) == 0:
        st.write("Object-level embedding file is not available for this project.")
        return

    n_cols = get_state().page_grid_settings.columns
    n_rows = get_state().page_grid_settings.rows

    metric = get_state().selected_metric
    if not metric:
        st.error("Metric not selected.")
        return

    if embedding_type not in get_state().reduced_embeddings:
        get_state().reduced_embeddings[embedding_type] = get_2d_embedding_data(
            get_state().project_paths.embeddings, embedding_type
        )

    if get_state().reduced_embeddings[embedding_type] is None:
        st.info("There is no 2D embedding file to display.")
    else:
        # Apply if a high level filter (class or annotator) is applied
        current_reduced_embedding = get_state().reduced_embeddings[embedding_type]
        reduced_embedding_filtered = current_reduced_embedding[
            current_reduced_embedding[Embedding2DSchema.identifier].isin(current_df[MetricSchema.identifier])
        ]
        selected_rows = render_plotly_events(reduced_embedding_filtered)
        if selected_rows is not None:
            current_df = current_df[
                current_df[MetricSchema.identifier].isin(selected_rows[Embedding2DSchema.identifier])
            ]

    chart = get_histogram(current_df, "score", metric.name)
    st.altair_chart(chart, use_container_width=True)
    subset = render_df_slicer(current_df, "score")

    showing_description = "images" if metric_scope == MetricScope.DATA_QUALITY else "labels"
    st.write(f"Interval contains {subset.shape[0]} of {current_df.shape[0]} {showing_description}")

    paginated_subset = render_pagination(subset, n_cols, n_rows, "score")

    form = bulk_tagging_form(metric_scope)

    if form and form.submitted:
        df = paginated_subset if form.level == BulkLevel.PAGE else subset
        action_bulk_tags(df, form.tags, form.action)

    if len(paginated_subset) == 0:
        st.error("No data in selected interval")
    else:
        cols: List = []
        similarity_expanders = []
        for i, (_, row) in enumerate(paginated_subset.iterrows()):
            if not cols:
                if i:
                    divider()
                cols = list(st.columns(n_cols))
                similarity_expanders.append(st.expander("Similarities", expanded=True))

            with cols.pop(0):
                build_card(embedding_information, i, row, similarity_expanders, metric_scope, metric)


def build_card(
    embedding_information: SimilaritiesFinder,
    card_no: int,
    row: pd.Series,
    similarity_expanders: list[DeltaGenerator],
    metric_scope: MetricScope,
    metric: MetricData,
):
    """
    Builds each sub card (the content displayed for each row in a csv file).
    """
    data_dir = get_state().project_paths.data

    identifier_parts = 4 if embedding_information.has_annotations else 3
    identifier = "_".join(str(row["identifier"]).split("_")[:identifier_parts])

    if embedding_information.type in [EmbeddingType.IMAGE, EmbeddingType.CLASSIFICATION]:
        button_name = "show similar images"
        image = load_or_fill_image(row, data_dir)
    elif embedding_information.type == EmbeddingType.OBJECT:
        button_name = "show similar objects"
        image = show_image_and_draw_polygons(row, data_dir)
    else:
        st.write(f"{embedding_information.type.value} card type is not defined in EmbeddingTypes")
        return

    st.image(image)

    # === Write scores and link to editor === #
    tags_row = row.copy()

    if "object_class" in tags_row and not pd.isna(tags_row["object_class"]):
        tags_row["label_class_name"] = tags_row["object_class"]
        tags_row.drop("object_class")
    tags_row[metric.name] = tags_row["score"]
    build_data_tags(tags_row, metric.name)

    if not pd.isnull(row["description"]):
        # Hacky way for now (with incorrect rounding)
        description = re.sub(r"(\d+\.\d{0,3})\d*", r"\1", row["description"])
        st.write(f"Description: {description}")

    multiselect_tag(row, "explorer", metric_scope)

    target_expander = similarity_expanders[card_no // get_state().page_grid_settings.columns]

    st.button(
        str(button_name),
        key=f"similarity_button_{row['identifier']}",
        on_click=show_similarities,
        args=(identifier, target_expander, embedding_information),
    )
