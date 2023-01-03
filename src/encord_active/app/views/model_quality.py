import streamlit as st

import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.components import sticky_header
from encord_active.app.common.state import (
    PREDICTIONS_FULL_CLASS_IDX,
    PREDICTIONS_LABEL_METRIC,
    PREDICTIONS_LABEL_METRIC_NAMES,
    PREDICTIONS_LABELS,
    PREDICTIONS_METRIC,
    PREDICTIONS_METRIC_META,
    get_state,
    setdefault,
)
from encord_active.app.common.state_hooks import use_lazy_state, use_memo
from encord_active.app.common.utils import setup_page
from encord_active.app.model_quality.settings import common_settings
from encord_active.app.model_quality.sub_pages import Page
from encord_active.lib.model_predictions.filters import (
    filter_labels_for_frames_wo_predictions,
    prediction_and_label_filtering,
)
from encord_active.lib.model_predictions.map_mar import compute_mAP_and_mAR


def model_quality(page: Page):
    def render():
        setup_page()

        if not reader.check_model_prediction_availability(get_state().project_paths.predictions):
            st.markdown(
                "# Missing Model Predictions\n"
                "This project does not have any imported predictions. "
                "Please refer to the "
                "[Importing Model Predictions](https://encord-active-docs.web.app/sdk/importing-model-predictions) "
                "section of the documentation to learn how to import your predictions."
            )
            return

        predictions_dir = get_state().project_paths.predictions
        metrics_dir = get_state().project_paths.metrics

        if not get_state().predictions.metric_datas:
            get_state().predictions.metric_datas = reader.get_prediction_metric_data(predictions_dir, metrics_dir)

        model_predictions = use_memo(
            lambda: reader.get_model_predictions(predictions_dir, get_state().predictions.metric_datas)
        )
        selected_class_idx = get_state().predictions.selected_classes

        if selected_class_idx is None or model_predictions is None:
            st.error("Couldn't load model predictions")
            return

        label_metric_data = setdefault(PREDICTIONS_LABEL_METRIC_NAMES, reader.get_label_metric_data, metrics_dir)
        setdefault(PREDICTIONS_LABELS, reader.get_labels, predictions_dir, label_metric_data)

        if st.session_state[PREDICTIONS_LABELS] is None:
            st.error("Couldn't load labels properly")
            return

        get_gt_matched, _ = use_lazy_state(lambda: reader.get_gt_matched(predictions_dir))
        st.session_state[PREDICTIONS_METRIC_META] = {
            "predictions": {m.name: m for m in get_state().predictions.metric_datas},
            "labels": {m.name: m for m in label_metric_data},
        }

        matched_gt = get_gt_matched()
        if not matched_gt:
            st.error("Couldn't match groung truths")
            return

        with sticky_header():
            common_settings()
            page.sidebar_options()

        (model_predictions, st.session_state[PREDICTIONS_LABELS], metrics, precisions,) = compute_mAP_and_mAR(
            model_predictions,
            st.session_state.get(PREDICTIONS_LABELS),  # type: ignore
            matched_gt,
            st.session_state.get(PREDICTIONS_FULL_CLASS_IDX),  # type: ignore
            iou_threshold=get_state().iou_threshold,
            ignore_unmatched_frames=get_state().ignore_frames_without_predictions,
        )

        # Sort predictions and labels according to selected metrics.
        pred_sort_column = st.session_state.get(PREDICTIONS_METRIC, get_state().predictions.metric_datas[0].name)
        sorted_model_predictions = model_predictions.sort_values([pred_sort_column], axis=0)

        label_sort_column = st.session_state.get(
            PREDICTIONS_LABEL_METRIC, st.session_state[PREDICTIONS_LABEL_METRIC_NAMES][0].name
        )
        st.session_state.sorted_labels = st.session_state[PREDICTIONS_LABELS].sort_values([label_sort_column], axis=0)

        if get_state().ignore_frames_without_predictions:
            labels = filter_labels_for_frames_wo_predictions(model_predictions, st.session_state.sorted_labels)
        else:
            labels = st.session_state.sorted_labels

        _labels, _metrics, _model_pred, _precisions = prediction_and_label_filtering(
            st.session_state.selected_class_idx, labels, metrics, sorted_model_predictions, precisions
        )
        page.build(model_predictions=_model_pred, labels=_labels, metrics=_metrics, precisions=_precisions)

    return render
