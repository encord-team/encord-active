import re
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from natsort import natsorted
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator
from streamlit_plotly_events import plotly_events

from encord_active.app.actions_page.export_filter import render_filter
from encord_active.app.common.components import build_data_tags, divider
from encord_active.app.common.components.annotator_statistics import (
    render_annotator_properties,
)
from encord_active.app.common.components.divider import divider
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
from encord_active.lib.common.image_utils import show_image_and_draw_polygons
from encord_active.lib.embeddings.dimensionality_reduction import get_2d_embedding_data
from encord_active.lib.embeddings.utils import (
    Embedding2DSchema,
    PointSchema2D,
    PointSelectionSchema,
    SimilaritiesFinder,
)
from encord_active.lib.metrics.metric import EmbeddingType
from encord_active.lib.metrics.utils import (
    IdentifierSchema,
    MetricData,
    MetricSchema,
    MetricScope,
    get_annotator_level_info,
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

        df = get_state().filtering_state.sorted_items
        if df is None:
            return None

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

        filtered_merged_metrics = get_state().filtering_state.merged_metrics
        if filtered_merged_metrics is not None:
            df = df[df.identifier.isin(filtered_merged_metrics.identifier)]

        return df

    def build(self, selected_df: DataFrame[MetricSchema], metric_scope: MetricScope):
        selected_metric = get_state().filtering_state.sort_by_metric
        if not selected_metric:
            return

        if selected_metric.meta.doc_url is None:
            st.markdown(f"### {selected_metric.meta.title}")
        else:
            st.markdown(f"### [{selected_metric.meta.title}]({selected_metric.meta.doc_url})")

        if selected_df.empty:
            return

        with st.expander("Dataset Properties", expanded=True):
            render_dataset_properties(selected_df)
        with st.expander("Annotator Statistics", expanded=False):
            render_annotator_properties(selected_df)

        fill_data_quality_window(selected_df, metric_scope, selected_metric)

    def render_view_options(self, *args):
        non_empty_metrics = [
            metric for metric in self.available_metrics if not load_metric_dataframe(metric, normalize=False).empty
        ]
        sorted_metrics = natsorted(non_empty_metrics, key=lambda i: i.name)

        metric_names = list(map(lambda i: i.name, sorted_metrics))

        if not metric_names:
            return

        col1, col2, col3, col4 = st.columns(4)
        selected_metric_name = col1.selectbox(
            "Sort by",
            metric_names,
            help="The data in the main view will be sorted by the selected metric. ",
        )
        if not selected_metric_name:
            return None
        sorting_order = col2.selectbox("Sort order", ["Ascending", "Descending"])

        metric_idx = metric_names.index(selected_metric_name)
        selected_metric = sorted_metrics[metric_idx]
        get_state().filtering_state.sort_by_metric = selected_metric

        df: DataFrame[MetricSchema] = load_metric_dataframe(selected_metric)
        df = df.sort_values(by="score", ascending=sorting_order == "Ascending").pipe(DataFrame[MetricSchema])
        get_state().filtering_state.sorted_items = df
        if df.shape[0] <= 0:
            return None

        class_set = natsorted(df[MetricSchema.object_class].dropna().unique().tolist())
        with col3:
            if len(class_set) > 0:
                get_state().filtering_state.selected_classes = st.multiselect("Filter by class", class_set)

        annotators = get_annotator_level_info(df)
        annotator_set = natsorted(annotators.keys())
        with col4:
            if len(annotator_set) > 0:
                get_state().filtering_state.selected_annotators = st.multiselect("Filter by annotator", annotator_set)

        render_filter()
        divider()
        super().render_common_settings(*args)


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
    should_select = UseState(True)
    selection = UseState[Optional[List[dict]]](None)

    fig = px.scatter(
        embedding_2d,
        x=Embedding2DSchema.x,
        y=Embedding2DSchema.y,
        color=Embedding2DSchema.label,
        title="2D embedding plot",
        template="plotly",
    )

    new_selection = plotly_events(fig, click_event=False, select_event=True)

    if new_selection != selection.value:
        should_select.set(True)
        selection.set(new_selection)

    if st.button("Reset selection"):
        should_select.set(False)

    if should_select.value and len(new_selection) > 0:
        return get_selected_rows(embedding_2d, new_selection)
    else:
        return None


def fill_data_quality_window(
    current_df: DataFrame[MetricSchema], metric_scope: MetricScope, selected_metric: MetricData
):
    meta = selected_metric.meta
    embedding_type = get_embedding_type(meta.annotation_type)
    embeddings_dir = get_state().project_paths.embeddings
    embedding_information = SimilaritiesFinder(
        embedding_type, embeddings_dir, num_of_neighbors=get_state().similarities_count
    )

    if (embedding_information.type == EmbeddingType.CLASSIFICATION) and len(embedding_information.collections) == 0:
        st.write("Image-level embedding file is not available for this project.")
        return
    if (embedding_information.type == EmbeddingType.OBJECT) and len(embedding_information.collections) == 0:
        st.write("Object-level embedding file is not available for this project.")
        return

    n_cols = get_state().page_grid_settings.columns
    n_rows = get_state().page_grid_settings.rows

    metric = get_state().filtering_state.sort_by_metric
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

    showing_description = "images" if metric_scope == MetricScope.DATA_QUALITY else "labels"
    st.write(f"Interval contains {current_df.shape[0]} of {current_df.shape[0]} {showing_description}")

    form = bulk_tagging_form(current_df.pipe(DataFrame[IdentifierSchema]))
    page_number = UseState(1)
    n_items = n_cols * n_rows
    paginated_subset = paginate_df(current_df, page_number.value, n_items)

    if form and form.submitted:
        df = paginated_subset if form.level == BulkLevel.PAGE else current_df
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

    last = len(current_df) // n_items + 1
    page_number.set(st.slider("Page", 1, last) if last > 1 else 1)


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
    identifier_parts = 4 if embedding_information.has_annotations else 3
    identifier = "_".join(str(row["identifier"]).split("_")[:identifier_parts])

    if embedding_information.type in [EmbeddingType.IMAGE, EmbeddingType.CLASSIFICATION]:
        button_name = "Show similar images"
    elif embedding_information.type == EmbeddingType.OBJECT:
        button_name = "Show similar objects"
    else:
        st.write(f"{embedding_information.type.value} card type is not defined in EmbeddingTypes")
        return

    image = show_image_and_draw_polygons(
        row, get_state().project_paths, draw_configurations=get_state().object_drawing_configurations
    )
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

    multiselect_tag(row, "explorer")

    target_expander = similarity_expanders[card_no // get_state().page_grid_settings.columns]

    st.button(
        str(button_name),
        key=f"similarity_button_{row['identifier']}",
        on_click=show_similarities,
        args=(identifier, target_expander, embedding_information),
    )
