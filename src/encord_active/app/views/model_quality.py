import streamlit as st

import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.components import sticky_header
from encord_active.app.common.components.tags.tag_creator import tag_creator
from encord_active.app.common.state import MetricNames, get_state
from encord_active.app.common.state_hooks import use_memo
from encord_active.app.common.utils import setup_page
from encord_active.app.model_quality.settings import (
    common_settings_classifications,
    common_settings_objects,
)
from encord_active.app.model_quality.sub_pages import Page
from encord_active.lib.charts.classification_metrics import (
    get_accuracy,
    get_confusion_matrix,
    get_precision_recall_f1,
)
from encord_active.lib.constants import DOCS_URL
from encord_active.lib.model_predictions.filters import (
    filter_labels_for_frames_wo_predictions,
    prediction_and_label_filtering,
    prediction_and_label_filtering_classification,
)
from encord_active.lib.model_predictions.map_mar import compute_mAP_and_mAR
from encord_active.lib.model_predictions.writer import MainPredictionType


def model_quality(page: Page):
    def render():
        setup_page()
        tag_creator()

        object_tab, classification_tab = st.tabs(["Objects", "Classifications"])

        with object_tab:
            if not reader.check_model_prediction_availability(
                get_state().project_paths.predictions / MainPredictionType.OBJECT.value
            ):
                st.markdown(
                    "## Missing model predictions for the objects\n"
                    "This project does not have any imported predictions for the objects. "
                    "Please refer to the "
                    f"[Importing Model Predictions]({DOCS_URL}/sdk/importing-model-predictions) "
                    "section of the documentation to learn how to import your predictions."
                )
            else:

                predictions_dir = get_state().project_paths.predictions / MainPredictionType.OBJECT.value
                metrics_dir = get_state().project_paths.metrics

                predictions_metric_datas = use_memo(
                    lambda: reader.get_prediction_metric_data(predictions_dir, metrics_dir)
                )
                label_metric_datas = use_memo(lambda: reader.get_label_metric_data(metrics_dir))
                model_predictions = use_memo(
                    lambda: reader.get_model_predictions(predictions_dir, predictions_metric_datas)
                )
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
                    st.error("Couldn't match ground truths")
                    return

                with sticky_header():
                    common_settings_objects()
                    page.sidebar_options()

                (matched_predictions, matched_labels, metrics, precisions,) = compute_mAP_and_mAR(
                    model_predictions,
                    labels,
                    matched_gt,
                    get_state().predictions.all_classes_objects,
                    iou_threshold=get_state().iou_threshold,
                    ignore_unmatched_frames=get_state().ignore_frames_without_predictions,
                )

                # Sort predictions and labels according to selected metrics.
                pred_sort_column = (
                    get_state().predictions.metric_datas.selected_prediction or predictions_metric_datas[0].name
                )
                sorted_model_predictions = matched_predictions.sort_values([pred_sort_column], axis=0)

                label_sort_column = get_state().predictions.metric_datas.selected_label or label_metric_datas[0].name
                sorted_labels = matched_labels.sort_values([label_sort_column], axis=0)

                if get_state().ignore_frames_without_predictions:
                    matched_labels = filter_labels_for_frames_wo_predictions(matched_predictions, sorted_labels)
                else:
                    matched_labels = sorted_labels

                _labels, _metrics, _model_pred, _precisions = prediction_and_label_filtering(
                    get_state().predictions.selected_classes_objects,
                    matched_labels,
                    metrics,
                    sorted_model_predictions,
                    precisions,
                )
                page.build(model_predictions=_model_pred, labels=_labels, metrics=_metrics, precisions=_precisions)

        with classification_tab:
            if not reader.check_model_prediction_availability(
                get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
            ):
                st.markdown(
                    "## Missing model predictions for the classifications\n"
                    "This project does not have any imported predictions for the classifications. "
                    "Please refer to the "
                    f"[Importing Model Predictions]({DOCS_URL}/sdk/importing-model-predictions) "
                    "section of the documentation to learn how to import your predictions."
                )
            else:

                predictions = reader.get_classification_predictions(
                    get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
                )
                labels = reader.get_classification_labels(
                    get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
                )

                with sticky_header():
                    common_settings_classifications()

                matched_predictions, matched_labels = prediction_and_label_filtering_classification(
                    get_state().predictions.selected_classes_classifications, labels, predictions
                )

                # --- THE FOLLOWINGS WILL BE MOVED INTO page.build() METHOD LATER ---

                # y_true, y_pred = list(predictions[reader.ClassificationLabelSchema.class_id]), list(
                #     labels[reader.ClassificationPredictionSchema.class_id]
                # )

                y_true, y_pred = list(matched_predictions[reader.ClassificationLabelSchema.class_id]), list(
                    matched_labels[reader.ClassificationPredictionSchema.class_id]
                )

                precision, recall, f1, _ = get_precision_recall_f1(y_true, y_pred)
                accuracy = get_accuracy(y_true, y_pred)

                col_acc, col_prec, col_rec, col_f1 = st.columns(4)
                col_acc.metric("Accuracy", f"{float(accuracy):.2f}")
                col_prec.metric("Mean Precision", f"{float(precision.mean()):.2f}")
                col_rec.metric("Mean Recall", f"{float(recall.mean()):.2f}")
                col_f1.metric("Mean F1", f"{float(f1.mean()):.2f}")

                sorted_class_ids = sorted([k for k in get_state().predictions.all_classes_classifications])
                class_names = [
                    get_state().predictions.all_classes_classifications[class_id]["name"]
                    for class_id in sorted_class_ids
                ]

                confusion_matrix = get_confusion_matrix(y_true, y_pred, class_names)
                st.plotly_chart(confusion_matrix)

    return render
