from typing import List, Optional

import pandas as pd
import streamlit as st
from pandera.typing import DataFrame

from encord_active.app.common.components import build_data_tags, divider
from encord_active.app.common.components.paginator import render_pagination
from encord_active.app.common.components.slicer import render_df_slicer
from encord_active.app.common.components.tags.bulk_tagging_form import (
    BulkLevel,
    action_bulk_tags,
    bulk_tagging_form,
)
from encord_active.app.common.components.tags.individual_tagging import multiselect_tag
from encord_active.app.common.state import get_state
from encord_active.lib.common.colors import Color
from encord_active.lib.common.image_utils import (
    draw_object,
    show_image_and_draw_polygons,
    show_image_with_predictions_and_label,
)
from encord_active.lib.model_predictions.reader import (
    ClassificationPredictionMatchSchema,
    ClassificationPredictionMatchSchemaWithClassNames,
    LabelMatchSchema,
    PredictionMatchSchema,
)
from encord_active.lib.project import ProjectFileStructure


def build_card_for_labels(
    label: pd.Series,
    predictions: DataFrame[PredictionMatchSchema],
    project_file_structure: ProjectFileStructure,
    label_color: Color = Color.RED,
):
    class_colors = {
        int(index): object.get("color", Color.RED)
        for index, object in get_state().predictions.all_classes_objects.items()
    }
    image = show_image_with_predictions_and_label(
        label,
        predictions,
        project_file_structure,
        label_color=label_color,
        class_colors=class_colors,
        draw_configurations=get_state().object_drawing_configurations,
    )
    st.image(image)
    multiselect_tag(label, "false_negatives")

    cls = get_state().predictions.all_classes_objects[str(label["class_id"])]["name"]
    label = label.copy()
    label["label_class_name"] = cls
    # === Write scores and link to editor === #
    build_data_tags(label, get_state().predictions.metric_datas.selected_label)


def build_card_for_predictions(row: pd.Series, project_file_structure: ProjectFileStructure, box_color=Color.GREEN):
    conf = get_state().object_drawing_configurations
    image = show_image_and_draw_polygons(row, project_file_structure, draw_configurations=conf, skip_object_hash=True)
    image = draw_object(image, row, draw_configuration=conf, color=box_color, with_box=True)
    st.image(image)
    multiselect_tag(row, "metric_view", is_predictions=True)

    # === Write scores and link to editor === #
    build_data_tags(row, get_state().predictions.metric_datas.selected_prediction)

    if row[PredictionMatchSchema.false_positive_reason] and not row[PredictionMatchSchema.is_true_positive]:
        st.write(f"Reason: {row[PredictionMatchSchema.false_positive_reason]}")


def build_card(
    row: pd.Series,
    predictions: Optional[DataFrame[PredictionMatchSchema]],
    project_file_structure: ProjectFileStructure,
    box_color: Color = Color.GREEN,
):
    if predictions is not None:
        build_card_for_labels(row, predictions, project_file_structure, box_color)
    else:
        build_card_for_predictions(row, project_file_structure, box_color)


def build_card_classifications(row: pd.Series, project_file_structure: ProjectFileStructure):
    image = show_image_and_draw_polygons(row, project_file_structure)
    st.image(image)
    multiselect_tag(row, "metric_view_classification", is_predictions=True)
    build_data_tags(row, get_state().predictions.metric_datas_classification.selected_prediction)

    if row[ClassificationPredictionMatchSchema.is_true_positive] == 0.0:
        st.markdown(f"Ground truth: `{row[ClassificationPredictionMatchSchemaWithClassNames.gt_class_name]}`")


def prediction_grid(
    project_file_structure: ProjectFileStructure,
    model_predictions: DataFrame[PredictionMatchSchema],
    labels: Optional[DataFrame[LabelMatchSchema]] = None,
    box_color: Color = Color.GREEN,
):
    use_labels = labels is not None
    if use_labels:
        df = labels
        additionals = model_predictions
        selected_metric = get_state().predictions.metric_datas.selected_label or ""
    else:
        df = model_predictions
        additionals = None
        selected_metric = get_state().predictions.metric_datas.selected_prediction or ""

    n_cols, n_rows = get_state().page_grid_settings.columns, get_state().page_grid_settings.rows
    subset = render_df_slicer(df, selected_metric)
    paginated_subset = render_pagination(subset, n_cols, n_rows, selected_metric)

    form = bulk_tagging_form(subset, is_predictions=True)
    if form and form.submitted:
        df = paginated_subset if form.level == BulkLevel.PAGE else subset
        action_bulk_tags(df, form.tags, form.action)

    if len(paginated_subset) == 0:
        st.error("No data in selected quality interval")
    else:
        cols: List = []
        for i, (_, row) in enumerate(paginated_subset.iterrows()):
            frame_additionals: Optional[DataFrame[PredictionMatchSchema]] = None
            if additionals is not None:
                frame_additionals = additionals[
                    additionals[PredictionMatchSchema.img_id] == row[LabelMatchSchema.img_id]
                ]

            if not cols:
                if i:
                    divider()
                cols = list(st.columns(n_cols))
            with cols.pop(0):
                build_card(row, frame_additionals, project_file_structure, box_color=box_color)


def prediction_grid_classifications(
    project_file_structure: ProjectFileStructure,
    model_predictions: DataFrame[ClassificationPredictionMatchSchemaWithClassNames],
):
    df = model_predictions
    selected_metric = get_state().predictions.metric_datas_classification.selected_prediction or ""

    n_cols, n_rows = get_state().page_grid_settings.columns, get_state().page_grid_settings.rows
    subset = render_df_slicer(df, selected_metric)
    paginated_subset = render_pagination(subset, n_cols, n_rows, selected_metric)

    form = bulk_tagging_form(subset, is_predictions=True)
    if form and form.submitted:
        df = paginated_subset if form.level == BulkLevel.PAGE else subset
        action_bulk_tags(df, form.tags, form.action)

    if len(paginated_subset) == 0:
        st.error("No data in the selected quality interval")
    else:
        cols: List = []
        for i, (_, row) in enumerate(paginated_subset.iterrows()):
            if not cols:
                if i:
                    divider()
                cols = list(st.columns(n_cols))
            with cols.pop(0):
                build_card_classifications(row, project_file_structure)
