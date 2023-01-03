from copy import deepcopy
from typing import Dict, List, Tuple

import streamlit as st

import encord_active.app.common.components as cst

# import encord_active.app.common.state as state
from encord_active.app.common.components.tags.tag_creator import tag_creator
from encord_active.app.common.state import (
    PREDICTIONS_FULL_CLASS_IDX,
    get_state,
    setdefault,
)
from encord_active.lib.model_predictions.reader import OntologyObjectJSON, get_class_idx


def common_settings():
    tag_creator()

    predictions_dir = get_state().project_paths.predictions
    class_idx: Dict[str, OntologyObjectJSON] = setdefault(PREDICTIONS_FULL_CLASS_IDX, get_class_idx, predictions_dir)
    col1, col2, col3 = st.columns([4, 4, 3])

    with col1:
        selected_classes = st.multiselect(
            "Select classes to include",
            list(class_idx.items()),
            format_func=lambda x: x[1]["name"],
            help="""
            With this selection, you can choose which classes to include in the main page.\n
            This acts as a filter, i.e. when nothing is selected all classes are included.
            """,
        )

    if not selected_classes:
        st.session_state.selected_class_idx = deepcopy(class_idx)
    else:
        st.session_state.selected_class_idx = dict(selected_classes)

    with col2:
        # IOU
        iou_threshold = st.slider(
            "Select an IOU threshold",
            min_value=0,
            max_value=100,
            value=50,
            help="The mean average precision (mAP) score is based on true positives and false positives. "
            "The IOU threshold determines how closely predictions need to match labels to be considered "
            "as true positives.",
        )
        get_state().iou_threshold = iou_threshold / 100

    with col3:
        st.write("")
        st.write("")
        # Ignore unmatched frames
        get_state().ignore_frames_without_predictions = st.checkbox(
            "Ignore frames without predictions",
            value=get_state().ignore_frames_without_predictions,
            help="Scores like mAP and mAR are effected negatively if there are frames in the dataset for which there "
            "exist no predictions. With this flag, you can ignore those.",
        )
