from copy import deepcopy

import streamlit as st

import encord_active.app.common.state as state
import encord_active.app.model_assertions.data as pred_data
from encord_active.app.common.components import sticky_header
from encord_active.app.common.utils import setup_page
from encord_active.app.model_assertions.map_mar import compute_mAP_and_mAR
from encord_active.app.model_assertions.settings import common_settings
from encord_active.app.model_assertions.sub_pages import Page


def model_assertions(page: Page):
    def render():
        setup_page()

        if not pred_data.check_model_prediction_availability():
            st.markdown(
                "# Missing Model Predictions\n"
                "This project does not have any imported predictions. "
                "Please refer to the "
                "[Importing Model Predictions](https://encord-active-docs.web.app/sdk/importing-model-predictions) "
                "section of the documentation to learn how to import your predictions."
            )
            return

        st.session_state.selected_class_idx = pred_data.get_class_idx()
        st.session_state.full_class_idx = deepcopy(pred_data.get_class_idx())
        (
            st.session_state.model_predictions,
            st.session_state.prediction_metric_names,
        ) = pred_data.get_model_predictions() or (None, None)
        if st.session_state.selected_class_idx is None:
            st.error("Couldn't load model predictions")
            return

        st.session_state.labels, st.session_state.label_metric_names = pred_data.get_labels() or (None, None)
        if st.session_state.labels is None:
            return

        st.session_state.gt_matched = pred_data.get_gt_matched()
        st.session_state.metric_meta = pred_data.get_metadata_files()

        with sticky_header():
            common_settings()
            page.sidebar_options()

        metrics, precisions, tps, fp_reasons, fns = compute_mAP_and_mAR(
            iou_threshold=st.session_state.get(state.IOU_THRESHOLD),
            ignore_unmatched_frames=st.session_state[state.IGNORE_FRAMES_WO_PREDICTIONS],
        )
        st.session_state.model_predictions["tps"] = tps.astype(float)
        st.session_state.model_predictions["fp_reason"] = fp_reasons["fp_reason"]
        st.session_state.labels["fns"] = fns

        # Sort predictions and labels according to selected metrics.
        pred_sort_column = st.session_state.get(state.PREDICTIONS_METRIC, st.session_state.prediction_metric_names[0])
        st.session_state.sorted_model_predictions = st.session_state.model_predictions.sort_values(
            [pred_sort_column], axis=0
        )

        label_sort_column = st.session_state.get(state.PREDICTIONS_LABEL_METRIC, st.session_state.label_metric_names[0])
        st.session_state.sorted_labels = st.session_state.labels.sort_values([label_sort_column], axis=0)

        if st.session_state[state.IGNORE_FRAMES_WO_PREDICTIONS]:
            labels = pred_data.filter_labels_for_frames_wo_predictions()
        else:
            labels = st.session_state.sorted_labels

        _labels, _metrics, _model_pred, _precisions = pred_data.prediction_and_label_filtering(
            labels, metrics, st.session_state.sorted_model_predictions, precisions
        )
        page.build(model_predictions=_model_pred, labels=_labels, metrics=_metrics, precisions=_precisions)

    return render
