import re
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple
from urllib import parse

import pandas as pd
import streamlit as st
from encord_active_components.components.explorer import (
    GalleryItem,
    GroupedTags,
    Metadata,
    Output,
    OutputAction,
    PaginationInfo,
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
    show_image_and_draw_polygons,
)
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

    def render_view_options(self):
        non_empty_metrics = list(filter(filter_none_empty_metrics, self.available_metrics))
        metric_data_options = natsorted(non_empty_metrics, key=lambda i: i.name)

        if not metric_data_options:
            return

        col1, col2, _ = st.columns([5, 3, 3])
        selected_metric = col1.selectbox(
            "Sort by",
            metric_data_options,
            format_func=lambda x: x.meta.title,
            help="The data in the main view will be sorted by the selected metric. ",
        )
        if not selected_metric:
            return None

        sorting_order = col2.selectbox("Sort order", ["Ascending", "Descending"])

        get_state().filtering_state.sort_by_metric = selected_metric

        get_state().filtering_state.merged_metrics = get_state().merged_metrics.sort_values(
            by=selected_metric.meta.title, ascending=sorting_order == "Ascending"
        )
        render_filter()


@st.cache_data
def cached_histogram(_df: pd.DataFrame, column_name: str, metric_name: Optional[str] = None):
    return get_histogram(_df, column_name, metric_name)


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

    chart = cached_histogram(current_df, "score", metric.name)
    st.altair_chart(chart, use_container_width=True)

    showing_description = "images" if metric_scope == MetricScope.DATA_QUALITY else "labels"
    st.write(f"Interval contains {current_df.shape[0]} of {current_df.shape[0]} {showing_description}")

    items = []

    merged_metrics = get_state().merged_metrics
    with_all_metrics = current_df[["identifier"]].join(merged_metrics, on="identifier", how="left").dropna(axis=1)

    for row in with_all_metrics.to_dict("records"):
        id_parts = 4 if embedding_information.has_annotations else 3
        identifier = row.pop("identifier")
        split_id = str(identifier).split("_")
        lr, du, *_ = split_id
        url = get_url(lr, du)
        if not url:
            continue

        items.append(
            GalleryItem(
                id="_".join(split_id[:id_parts]),
                url=url,
                editUrl=row.pop("url"),
                tags=to_grouped_tags(row.pop("tags")),
                metadata=Metadata(
                    labelClass=row.pop("object_class", None),
                    annotator=row.pop("annotator", None),
                    metrics=row,
                ),
            )
        )

    output_state = UseState[Optional[Output]](None)
    output = explorer(items, to_grouped_tags(Tags().all()))
    print(output)
    if output and output != output_state:
        output_state.set(output)
        action, payload = output


def get_url(lr_hash: str, du_hash: str):
    for data_unit in get_state().project_paths.label_row_structure(lr_hash).iter_data_unit():
        if data_unit.hash == du_hash:
            return "http://localhost:8000/" + parse.quote(
                data_unit.path.relative_to(get_state().target_path).as_posix()
            )


@st.cache_data
def get_image(id: str, _row):
    return show_image_and_draw_polygons(
        _row, get_state().project_paths, draw_configurations=get_state().object_drawing_configurations
    )


def to_grouped_tags(tags: List[Tag]):
    grouped_tags = GroupedTags(data=[], label=[])

    for name, scope in tags:
        if scope == TagScope.DATA:
            grouped_tags[scope.lower()].append(name)
        elif scope == TagScope.LABEL:
            grouped_tags["label"].append(name)

    return grouped_tags

    # with grid_container:
    #     if form and form.submitted:
    #         df = paginated_subset if form.level == BulkLevel.PAGE else current_df
    #         action_bulk_tags(df, form.tags, form.action)
    #
    #     if len(paginated_subset) == 0:
    #         st.error("No data in selected interval")
    #     else:
    #         cols: List = []
    #         similarity_expanders = []
    #         for i, (_, row) in enumerate(paginated_subset.iterrows()):
    #             if not cols:
    #                 if i:
    #                     divider()
    #                 cols = list(st.columns(n_cols))
    #                 similarity_expanders.append(st.expander("Similarities", expanded=True))


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
