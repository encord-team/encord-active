from typing import List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import encord_active.app.model_assertions.components.utils as cutils
from encord_active.app.common import state
from encord_active.app.common.colors import Color, hex_to_rgb
from encord_active.app.common.components import build_data_tags
from encord_active.app.common.utils import (
    build_pagination,
    get_df_subset,
    get_geometries,
    load_or_fill_image,
)


def __show_image_and_draw_polygons_plus_prediction(row: pd.Series, box_color: Color = Color.GREEN) -> np.ndarray:
    isClosed = True
    thickness = 5

    # Label polygons
    image = load_or_fill_image(row)
    img_h, img_w = image.shape[:2]
    for color, geometry in get_geometries(row, img_h=img_h, img_w=img_w, skip_object_hash=True):
        image = cv2.polylines(image, [geometry], isClosed, hex_to_rgb(color), thickness)

    # Prediction polygon
    box = cutils.get_bbox(row)
    pred_color = hex_to_rgb(box_color.value)
    image = cv2.polylines(image, [box], isClosed, pred_color, thickness // 2, lineType=cv2.LINE_8)
    image = cutils.draw_mask(row, image, color=box_color)

    return image


def __build_card(row: pd.Series, st_col: DeltaGenerator, box_color: Color = Color.GREEN):
    with st_col:
        image = __show_image_and_draw_polygons_plus_prediction(row, box_color=box_color)
        st.image(image)

        # === Write scores and link to editor === #
        build_data_tags(row, st.session_state.predictions_metric)

        if row["fp_reason"] and not row["tps"] == 1.0:
            st.write(f"Reason: {row['fp_reason']}")


def metric_view(
    df: pd.DataFrame,
    box_color: Color = Color.GREEN,
):
    if state.METRIC_VIEW_PAGE_NUMBER not in st.session_state:
        st.session_state[state.METRIC_VIEW_PAGE_NUMBER] = 1

    n_cols, n_rows = int(st.session_state[state.MAIN_VIEW_COLUMN_NUM]), int(st.session_state[state.MAIN_VIEW_ROW_NUM])
    selected_metric = st.session_state.get(state.PREDICTIONS_METRIC, "")
    subset = get_df_subset(df, selected_metric)
    paginated_subset = build_pagination(subset, n_cols, n_rows, selected_metric)

    if len(paginated_subset) == 0:
        st.error("No data in selected quality interval")
    else:
        # === Fill in the container === #
        cols: List = []
        for i, row in paginated_subset.iterrows():
            if not cols:
                cols = list(st.columns(n_cols))
            __build_card(row, cols.pop(0), box_color=box_color)
