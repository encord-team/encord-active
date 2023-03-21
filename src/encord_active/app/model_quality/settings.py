from copy import deepcopy

import streamlit as st

from encord_active.app.common.state import State, get_state
from encord_active.lib.model_predictions.reader import get_class_idx
from encord_active.lib.model_predictions.writer import MainPredictionType

# import encord_active.app.common.state as state


def common_settings_objects():

    if not get_state().predictions.all_classes_objects:
        get_state().predictions.all_classes_objects = get_class_idx(
            get_state().project_paths.predictions / MainPredictionType.OBJECT.value
        )

    all_classes = get_state().predictions.all_classes_objects
    col1, col2, col3 = st.columns([4, 4, 3])

    with col1:
        selected_classes = st.multiselect(
            "Filter by class",
            list(all_classes.items()),
            format_func=lambda x: x[1]["name"],
            help="""
            With this selection, you can choose which classes to include in the performance metrics calculations.\n
            This acts as a filter, i.e. when nothing is selected all classes are included.
            """,
        )

    get_state().predictions.selected_classes_objects = dict(selected_classes) or deepcopy(all_classes)

    with col2:
        # IOU
        get_state().iou_threshold = st.slider(
            "Select an IOU threshold",
            min_value=0.0,
            max_value=1.0,
            value=State.iou_threshold,
            help="The mean average precision (mAP) score is based on true positives and false positives. "
            "The IOU threshold determines how closely predictions need to match labels to be considered "
            "as true positives.",
        )

    with col3:
        st.write("")
        st.write("")
        # Ignore unmatched frames
        get_state().ignore_frames_without_predictions = st.checkbox(
            "Ignore frames without predictions",
            value=State.ignore_frames_without_predictions,
            help="Scores like mAP and mAR are effected negatively if there are frames in the dataset for which there "
            "exist no predictions. With this flag, you can ignore those.",
        )


def common_settings_classifications():
    if not get_state().predictions.all_classes_classifications:
        get_state().predictions.all_classes_classifications = get_class_idx(
            get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
        )

    all_classes = get_state().predictions.all_classes_classifications
    selected_classes = st.multiselect(
        "Filter by class",
        list(all_classes.items()),
        format_func=lambda x: x[1]["name"],
        help="""
        With this selection, you can choose which classes to include in the performance metrics calculations.
        This acts as a filter, i.e. when nothing is selected all classes are included.
        Performance metrics will be automatically updated according to the chosen classes.
        """,
    )

    get_state().predictions.selected_classes_classifications = dict(selected_classes) or deepcopy(all_classes)
