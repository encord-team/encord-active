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
    get_precision_recall_graph,
)
from encord_active.lib.charts.metric_importance import create_metric_importance_charts
from encord_active.lib.constants import DOCS_URL
from encord_active.lib.model_predictions.classification_metrics import (
    match_predictions_and_labels,
)
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
        metrics_dir = get_state().project_paths.metrics

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

                predictions_metric_datas = use_memo(
                    lambda: reader.get_prediction_metric_data(predictions_dir, metrics_dir)
                )
                label_metric_datas = use_memo(lambda: reader.get_label_metric_data(metrics_dir))
                model_predictions = use_memo(
                    lambda: reader.get_model_predictions(
                        predictions_dir, predictions_metric_datas, MainPredictionType.OBJECT
                    )
                )
                labels = use_memo(
                    lambda: reader.get_labels(predictions_dir, label_metric_datas, MainPredictionType.OBJECT)
                )

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

                (predictions_filtered, labels_filtered, metrics, precisions,) = compute_mAP_and_mAR(
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
                sorted_model_predictions = predictions_filtered.sort_values([pred_sort_column], axis=0)

                label_sort_column = get_state().predictions.metric_datas.selected_label or label_metric_datas[0].name
                sorted_labels = labels_filtered.sort_values([label_sort_column], axis=0)

                if get_state().ignore_frames_without_predictions:
                    labels_filtered = filter_labels_for_frames_wo_predictions(predictions_filtered, sorted_labels)
                else:
                    labels_filtered = sorted_labels

                _labels, _metrics, _model_pred, _precisions = prediction_and_label_filtering(
                    get_state().predictions.selected_classes_objects,
                    labels_filtered,
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
                predictions_dir_classification = (
                    get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
                )

                predictions = reader.get_classification_predictions(
                    get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
                )
                labels = reader.get_classification_labels(
                    get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
                )

                predictions_metric_datas = use_memo(
                    lambda: reader.get_prediction_metric_data(predictions_dir_classification, metrics_dir)
                )
                label_metric_datas = use_memo(lambda: reader.get_label_metric_data(metrics_dir))
                model_predictions = use_memo(
                    lambda: reader.get_model_predictions(
                        predictions_dir_classification, predictions_metric_datas, MainPredictionType.CLASSIFICATION
                    )
                )
                labels_all = use_memo(
                    lambda: reader.get_labels(
                        predictions_dir_classification, label_metric_datas, MainPredictionType.CLASSIFICATION
                    )
                )

                get_state().predictions.metric_datas_classification = MetricNames(
                    predictions={m.name: m for m in predictions_metric_datas},
                )

                if model_predictions is None:
                    st.error("Couldn't load model predictions")
                    return

                if labels_all is None:
                    st.error("Couldn't load labels properly")
                    return

                with sticky_header():
                    common_settings_classifications()

                model_predictions_matched = match_predictions_and_labels(model_predictions, labels_all)

                (
                    predictions_filtered,
                    labels_filtered,
                    model_predictions_matched_filtered,
                ) = prediction_and_label_filtering_classification(
                    get_state().predictions.selected_classes_classifications,
                    labels,
                    predictions,
                    model_predictions_matched,
                )

                # --- THE FOLLOWINGS WILL BE MOVED INTO page.build() METHOD LATER ---

                y_true, y_pred = (
                    list(labels_filtered[reader.ClassificationLabelSchema.class_id]),
                    list(predictions_filtered[reader.ClassificationPredictionSchema.class_id]),
                )

                class_names = sorted(list(set(y_true).union(y_pred)))

                precision, recall, f1, support = get_precision_recall_f1(y_true, y_pred)
                accuracy = get_accuracy(y_true, y_pred)

                # PERFORMANCE METRICS SUMMARY

                col_acc, col_prec, col_rec, col_f1 = st.columns(4)
                col_acc.metric("Accuracy", f"{float(accuracy):.2f}")
                col_prec.metric(
                    "Mean Precision",
                    f"{float(precision.mean()):.2f}",
                    help="Average of precision scores of all classes",
                )
                col_rec.metric(
                    "Mean Recall", f"{float(recall.mean()):.2f}", help="Average of recall scores of all classes"
                )
                col_f1.metric("Mean F1", f"{float(f1.mean()):.2f}", help="Average of F1 scores of all classes")

                # METRIC IMPORTANCE
                if model_predictions.shape[0] > 60_000:  # Computation are heavy so allow computing for only a subset.
                    num_samples = st.slider(
                        "Number of samples",
                        min_value=1,
                        max_value=len(model_predictions),
                        step=max(1, (len(model_predictions) - 1) // 100),
                        value=max((len(model_predictions) - 1) // 2, 1),
                        help="To avoid too heavy computations, we subsample the data at random to the selected size, "
                        "computing importance values.",
                    )
                    if num_samples < 100:
                        st.warning(
                            "Number of samples is too low to compute reliable index importances. "
                            "We recommend using at least 100 samples.",
                        )
                else:
                    num_samples = model_predictions.shape[0]

                metric_columns = list(get_state().predictions.metric_datas_classification.predictions.keys())
                metric_importance_chart = create_metric_importance_charts(
                    model_predictions_matched_filtered,
                    metric_columns=metric_columns,
                    num_samples=num_samples,
                    prediction_type=MainPredictionType.CLASSIFICATION,
                )
                st.altair_chart(metric_importance_chart, use_container_width=True)

                col1, col2 = st.columns(2)

                # CONFUSION MATRIX
                confusion_matrix = get_confusion_matrix(y_true, y_pred, class_names)
                col1.plotly_chart(confusion_matrix, use_container_width=True)

                # PRECISION_RECALL BARS
                pr_graph = get_precision_recall_graph(precision, recall, class_names)
                col2.plotly_chart(pr_graph, use_container_width=True)

                # In order to plot ROC curve, we need confidences for the ground
                # truth label. Currently, predictions.pkl file only has confidence
                # value for the predicted class.
                # roc_graph = get_roc_curve(y_true, y_prob)

                pr_graph = get_precision_recall_graph(precision, recall, class_names)
                st.plotly_chart(pr_graph)

                # In order to plot ROC curve, we need confidences for the ground
                # truth label. Currently, predictions.pkl file only has confidence
                # value for the predicted class.
                # roc_graph = get_roc_curve(y_true, y_prob)

                if model_predictions.shape[0] > 60_000:  # Computation are heavy so allow computing for only a subset.
                    num_samples = st.slider(
                        "Number of samples",
                        min_value=1,
                        max_value=len(model_predictions),
                        step=max(1, (len(model_predictions) - 1) // 100),
                        value=max((len(model_predictions) - 1) // 2, 1),
                        help="To avoid too heavy computations, we subsample the data at random to the selected size, "
                        "computing importance values.",
                    )
                    if num_samples < 100:
                        st.warning(
                            "Number of samples is too low to compute reliable index importances. "
                            "We recommend using at least 100 samples.",
                        )
                else:
                    num_samples = model_predictions.shape[0]

                get_state().predictions.metric_datas = MetricNames(
                    predictions={m.name: m for m in predictions_metric_datas},
                )
                metric_columns = list(get_state().predictions.metric_datas.predictions.keys())
                metric_importance_chart = create_metric_importance_charts(
                    model_predictions_matched,
                    metric_columns=metric_columns,
                    num_samples=num_samples,
                    prediction_type=MainPredictionType.CLASSIFICATION,
                )
                st.altair_chart(metric_importance_chart, use_container_width=True)

                col1, col2 = st.columns(2)

                # CONFUSION MATRTIX
                confusion_matrix = get_confusion_matrix(y_true, y_pred, class_names)
                col1.plotly_chart(confusion_matrix, use_container_width=True)

                # PRECISION_RECALL BARS
                pr_graph = get_precision_recall_graph(precision, recall, class_names)
                col2.plotly_chart(pr_graph, use_container_width=True)

                # In order to plot ROC curve, we need confidences for the ground
                # truth label. Currently, predictions.pkl file only has confidence
                # value for the predicted class.
                # roc_graph = get_roc_curve(y_true, y_prob)

    return render
