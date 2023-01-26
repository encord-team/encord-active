import streamlit as st

import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.components import sticky_header
from encord_active.app.common.state import MetricNames, get_state
from encord_active.app.common.state_hooks import use_memo
from encord_active.app.common.utils import setup_page
from encord_active.app.model_quality.settings import common_settings
from encord_active.app.model_quality.sub_pages import Page
from encord_active.lib.constants import DOCS_URL
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
                f"[Importing Model Predictions]({DOCS_URL}/sdk/importing-model-predictions) "
                "section of the documentation to learn how to import your predictions."
            )
            return

        predictions_dir = get_state().project_paths.predictions
        metrics_dir = get_state().project_paths.metrics

        predictions_metric_datas = use_memo(lambda: reader.get_prediction_metric_data(predictions_dir, metrics_dir))
        label_metric_datas = use_memo(lambda: reader.get_label_metric_data(metrics_dir))
        model_predictions = use_memo(lambda: reader.get_model_predictions(predictions_dir, predictions_metric_datas))
        labels = use_memo(lambda: reader.get_labels(predictions_dir, label_metric_datas))

        if model_predictions is None:
            st.error("Couldn't load model predictions")
            return

        if labels is None:
            st.error("Couldn't load labels properly")
            return

        matched_gt = use_memo(lambda: reader.get_gt_matched(predictions_dir))
        get_state().predictions.metric_datas = MetricNames(
            predictions={m.name: m for m in predictions_metric_datas},
            labels={m.name: m for m in label_metric_datas},
        )

        if not matched_gt:
            st.error("Couldn't match groung truths")
            return

        with sticky_header():
            common_settings()
            page.sidebar_options()

        (matched_predictions, matched_labels, metrics, precisions,) = compute_mAP_and_mAR(
            model_predictions,
            labels,
            matched_gt,
            get_state().predictions.all_classes,
            iou_threshold=get_state().iou_threshold,
            ignore_unmatched_frames=get_state().ignore_frames_without_predictions,
        )

        # Sort predictions and labels according to selected metrics.
        pred_sort_column = get_state().predictions.metric_datas.selected_predicion or predictions_metric_datas[0].name
        sorted_model_predictions = matched_predictions.sort_values([pred_sort_column], axis=0)

        label_sort_column = get_state().predictions.metric_datas.selected_label or label_metric_datas[0].name
        sorted_labels = matched_labels.sort_values([label_sort_column], axis=0)

        if get_state().ignore_frames_without_predictions:
            matched_labels = filter_labels_for_frames_wo_predictions(matched_predictions, sorted_labels)
        else:
            matched_labels = sorted_labels

        _labels, _metrics, _model_pred, _precisions = prediction_and_label_filtering(
            get_state().predictions.selected_classes, matched_labels, metrics, sorted_model_predictions, precisions
        )
        page.build(model_predictions=_model_pred, labels=_labels, metrics=_metrics, precisions=_precisions)

    return render
