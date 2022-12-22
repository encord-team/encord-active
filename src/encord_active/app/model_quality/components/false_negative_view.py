from typing import List, Optional

import cv2
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import encord_active.app.model_quality.components.utils as cutils
from encord_active.app.common import state
from encord_active.app.common.components import build_data_tags
from encord_active.app.common.components.paginator import render_pagination
from encord_active.app.common.components.slicer import render_df_slicer
from encord_active.app.common.components.tags.bulk_tagging_form import (
    BulkLevel,
    action_bulk_tags,
    bulk_tagging_form,
)
from encord_active.app.common.components.tags.individual_tagging import multiselect_tag
from encord_active.app.common.utils import load_or_fill_image
from encord_active.lib.common.colors import Color, hex_to_rgb
from encord_active.lib.metrics.load_metrics import MetricScope


def __show_image_and_fn(
    label: pd.Series,
    predictions: pd.DataFrame,
    box_color: Color = Color.RED,
    mask_opacity=0.5,
):
    """
    :param label: The csv row of the false-negative label to display.
    :param predictions: All the predictions on the same image with the samme predicted class.
    :param box_color: The hex color to use when drawing the prediction.
    """
    isClosed = True
    thickness = 5

    image = load_or_fill_image(label)

    # Draw predictions
    for _, pred in predictions.iterrows():
        pred_color: str = st.session_state.full_class_idx[str(pred["class_id"])]["color"]
        image = cutils.draw_mask(pred, image, mask_opacity=mask_opacity, color=pred_color)

    # Draw label
    label_color = hex_to_rgb(box_color.value)
    image = cv2.polylines(image, [cutils.get_bbox(label)], isClosed, label_color, thickness, lineType=cv2.LINE_8)
    image = cutils.draw_mask(label, image, mask_opacity=mask_opacity, color=box_color)

    st.image(image)


def __build_card(
    label: pd.Series,
    predictions: pd.DataFrame,
    st_col: DeltaGenerator,
    box_color: Color = Color.RED,
):
    with st_col:
        __show_image_and_fn(label, predictions, box_color=box_color)
        multiselect_tag(label, "false_negatives", MetricScope.MODEL_QUALITY)

        cls = st.session_state.full_class_idx[str(label["class_id"])]["name"]
        label = label.copy()
        label["label_class_name"] = cls
        # === Write scores and link to editor === #
        build_data_tags(label, st.session_state[state.PREDICTIONS_LABEL_METRIC])


def false_negative_view(false_negatives, model_predictions, color: Color):
    if state.FALSE_NEGATIVE_VIEW_PAGE_NUMBER not in st.session_state:
        st.session_state[state.FALSE_NEGATIVE_VIEW_PAGE_NUMBER] = 1

    n_cols, n_rows = int(st.session_state[state.MAIN_VIEW_COLUMN_NUM]), int(st.session_state[state.MAIN_VIEW_ROW_NUM])
    selected_metric: Optional[str] = st.session_state.get(state.PREDICTIONS_LABEL_METRIC)
    if not selected_metric:
        return

    subset = render_df_slicer(false_negatives, selected_metric)
    paginated_subset = render_pagination(subset, n_cols, n_rows, selected_metric)

    form = bulk_tagging_form(MetricScope.MODEL_QUALITY)

    if form and form.submitted:
        df = paginated_subset if form.level == BulkLevel.PAGE else subset
        action_bulk_tags(df, form.tags, form.action)

    if len(paginated_subset) == 0:
        st.error("No data in selected quality interval")
    else:
        # === Fill in the container === #
        cols: List = []
        for i, label in paginated_subset.iterrows():
            if not cols:
                cols = list(st.columns(n_cols))

            frame_preds = model_predictions[model_predictions["img_id"] == label["img_id"]]
            __build_card(label, frame_preds, cols.pop(0), box_color=color)
