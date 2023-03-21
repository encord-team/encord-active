from pathlib import Path

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
from encord_active.lib.model_predictions.classification_metrics import (
    match_predictions_and_labels,
)
from encord_active.lib.model_predictions.filters import (
    filter_labels_for_frames_wo_predictions,
    prediction_and_label_filtering,
    prediction_and_label_filtering_classification,
)
from encord_active.lib.model_predictions.map_mar import compute_mAP_and_mAR
from encord_active.lib.model_predictions.reader import (
    ClassificationLabelSchema,
    ClassificationPredictionMatchSchemaWithClassNames,
    ClassificationPredictionSchema,
)
from encord_active.lib.model_predictions.writer import MainPredictionType


def model_quality(page: Page):
    def _get_object_model_quality_data(metrics_dir: Path):
        if not reader.check_model_prediction_availability(
            get_state().project_paths.predictions / MainPredictionType.OBJECT.value
        ):
            return False, None, None, None, None

        predictions_dir = get_state().project_paths.predictions / MainPredictionType.OBJECT.value

        predictions_metric_datas = use_memo(lambda: reader.get_prediction_metric_data(predictions_dir, metrics_dir))
        label_metric_datas = use_memo(lambda: reader.get_label_metric_data(metrics_dir))
        model_predictions = use_memo(
            lambda: reader.get_model_predictions(predictions_dir, predictions_metric_datas, MainPredictionType.OBJECT)
        )
        labels = use_memo(lambda: reader.get_labels(predictions_dir, label_metric_datas, MainPredictionType.OBJECT))

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
        pred_sort_column = get_state().predictions.metric_datas.selected_prediction or predictions_metric_datas[0].name
        sorted_model_predictions = predictions_filtered.sort_values([pred_sort_column], axis=0)

        label_sort_column = get_state().predictions.metric_datas.selected_label or label_metric_datas[0].name
        sorted_labels = labels_filtered.sort_values([label_sort_column], axis=0)

        if get_state().ignore_frames_without_predictions:
            labels_filtered = filter_labels_for_frames_wo_predictions(predictions_filtered, sorted_labels)
        else:
            labels_filtered = sorted_labels

        object_labels, object_metrics, object_model_pred, object_precisions = prediction_and_label_filtering(
            get_state().predictions.selected_classes_objects,
            labels_filtered,
            metrics,
            sorted_model_predictions,
            precisions,
        )

        return True, object_labels, object_metrics, object_model_pred, object_precisions

    def _get_classification_model_quality_data(metrics_dir: Path):
        if not reader.check_model_prediction_availability(
            get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
        ):
            return False, None, None, None

        predictions_dir_classification = get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value

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
            page.sidebar_options_classifications()

        model_predictions_matched = match_predictions_and_labels(model_predictions, labels_all)

        (
            labels_filtered,
            predictions_filtered,
            model_predictions_matched_filtered,
        ) = prediction_and_label_filtering_classification(
            get_state().predictions.selected_classes_classifications,
            get_state().predictions.all_classes_classifications,
            labels,
            predictions,
            model_predictions_matched,
        )

        img_id_intersection = list(
            set(labels_filtered[ClassificationLabelSchema.img_id]).intersection(
                set(predictions_filtered[ClassificationPredictionSchema.img_id])
            )
        )
        labels_filtered_intersection = labels_filtered[
            labels_filtered[ClassificationLabelSchema.img_id].isin(img_id_intersection)
        ]
        predictions_filtered_intersection = predictions_filtered[
            predictions_filtered[ClassificationPredictionSchema.img_id].isin(img_id_intersection)
        ]

        y_true, y_pred = (
            list(labels_filtered_intersection[ClassificationLabelSchema.class_id]),
            list(predictions_filtered_intersection[ClassificationPredictionSchema.class_id]),
        )

        return (
            True,
            y_true,
            y_pred,
            model_predictions_matched_filtered.copy()[
                model_predictions_matched_filtered[ClassificationPredictionMatchSchemaWithClassNames.img_id].isin(
                    img_id_intersection
                )
            ],
        )

    def render():
        setup_page()
        tag_creator()

        """
        Note: Streamlit tabs should be initialized here for two reasons:
        1. Selected tab will be same for each page in the model quality tabs. Otherwise, if we create tabs inside the
        pages, selected tab will be reset in each click.
        2. Common top bar sticker includes filtering, so their result should be reflected in each page. Otherwise, it
        would be hard to get filters from other pages.
        """

        object_tab, classification_tab = st.tabs(["Objects", "Classifications"])
        metrics_dir = get_state().project_paths.metrics

        with object_tab:
            (
                object_predictions_exist,
                object_labels,
                object_metrics,
                object_model_pred,
                object_precisions,
            ) = _get_object_model_quality_data(metrics_dir)

        with classification_tab:

            (
                classification_predictions_exist,
                classification_labels,
                classification_pred,
                classification_model_predictions_matched,
            ) = _get_classification_model_quality_data(metrics_dir)

        page.build(
            object_predictions_exist,
            classification_predictions_exist,
            object_tab,
            classification_tab,
            object_model_predictions=object_model_pred,
            object_labels=object_labels,
            object_metrics=object_metrics,
            object_precisions=object_precisions,
            classification_labels=classification_labels,
            classification_pred=classification_pred,
            classification_model_predictions_matched=classification_model_predictions_matched,
        )

    return render
