from copy import deepcopy

import streamlit as st

import encord_active.app.common.components as cst
import encord_active.app.common.state as state


def common_settings():
    class_idx = st.session_state.full_class_idx
    col1, col2, col3 = st.columns([4, 4, 3])

    with col1:
        selected_classes = cst.multiselect_with_all_option(
            "Select classes to include",
            list(map(lambda x: x["name"], class_idx.values())),
            key=state.CLASS_SELECTION,
            help="With this selection, you can choose which classes to include in the main page.",
        )

    if "All" in selected_classes:
        st.session_state.selected_class_idx = deepcopy(class_idx)
    else:
        st.session_state.selected_class_idx = {k: v for k, v in class_idx.items() if v["name"] in selected_classes}

    with col2:
        # IOU
        iou_threshold = st.slider(
            "Select an IOU threshold",
            min_value=0,
            max_value=100,
            value=50,
            key=state.IOU_THRESHOLD_,
            help="The mean average precision (mAP) score is based on true positives and false positives. "
            "The IOU threshold determines how closely predictions need to match labels to be considered "
            "as true positives.",
        )
        st.session_state[state.IOU_THRESHOLD] = iou_threshold / 100

    with col3:
        st.write("")
        st.write("")
        # Ignore unmatched frames
        st.checkbox(
            "Ignore frames without predictions",
            key=state.IGNORE_FRAMES_WO_PREDICTIONS,
            help="Scores like mAP and mAR are effected negatively if there are frames in the dataset for which there "
            "exist no predictions. With this flag, you can ignore those.",
        )
