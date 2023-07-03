from typing import List, Optional, cast

import altair as alt
import streamlit as st
from encord_active_components.components.explorer import explorer
from loguru import logger
from pandera.typing import DataFrame

from encord_active.app.actions_page.export_filter import render_filter
from encord_active.app.auth.jwt import get_auth_token
from encord_active.app.common.state import MetricNames, get_state
from encord_active.app.model_quality.prediction_type_builder import (
    MetricType,
    ModelQualityPage,
    PredictionTypeBuilder,
)
from encord_active.lib.charts.classification_metrics import (
    get_accuracy,
    get_confusion_matrix,
    get_precision_recall_f1,
    get_precision_recall_graph,
)
from encord_active.lib.charts.performance_by_metric import performance_rate_by_metric
from encord_active.lib.charts.scopes import PredictionMatchScope
from encord_active.lib.model_predictions.filters import (
    prediction_and_label_filtering_classification,
)
from encord_active.lib.model_predictions.reader import (
    check_model_prediction_availability,
    match_predictions_and_labels,
    read_class_idx,
)
from encord_active.lib.model_predictions.types import (
    ClassificationLabelSchema,
    ClassificationOutcomeType,
    ClassificationPredictionMatchSchemaWithClassNames,
    ClassificationPredictionSchema,
    PredictionsFilters,
)
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.premium.model import TextQuery
from encord_active.lib.premium.querier import Querier
from encord_active.server.settings import Env, get_settings


class ClassificationTypeBuilder(PredictionTypeBuilder):
    title = "Classification"

    def __init__(self):
        self._explorer_outcome_type = ClassificationOutcomeType.CORRECT_CLASSIFICATIONS
        self._labels: Optional[List] = None
        self._predictions: Optional[List] = None
        self._model_predictions: Optional[DataFrame[ClassificationPredictionMatchSchemaWithClassNames]] = None

    def load_data(self, page_mode: ModelQualityPage) -> bool:
        self.page_mode = page_mode
        predictions_metric_datas, label_metric_datas, model_predictions, labels = self._read_prediction_files(
            MainPredictionType.CLASSIFICATION, project_path=get_state().project_paths.project_dir.as_posix()
        )

        if model_predictions is None:
            st.error("Couldn't load model predictions")
            return False

        if labels is None:
            st.error("Couldn't load labels properly")
            return False

        get_state().predictions.metric_datas_classification = MetricNames(
            predictions={m.name: m for m in predictions_metric_datas},
        )

        self.display_settings()

        model_predictions = cast(
            DataFrame[ClassificationPredictionSchema],
            model_predictions.sort_values(by=ClassificationPredictionSchema.img_id).reset_index(drop=True),
        )
        labels = cast(
            DataFrame[ClassificationLabelSchema],
            labels.sort_values(by=ClassificationLabelSchema.img_id).reset_index(drop=True),
        )

        model_predictions_matched = match_predictions_and_labels(model_predictions, labels)

        (
            labels_filtered,
            predictions_filtered,
            model_predictions_matched_filtered,
        ) = prediction_and_label_filtering_classification(
            get_state().predictions.selected_classes_classifications,
            get_state().predictions.all_classes_classifications,
            labels,
            model_predictions,
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

        self._labels, self._predictions = (
            list(labels_filtered_intersection[ClassificationLabelSchema.class_id]),
            list(predictions_filtered_intersection[ClassificationPredictionSchema.class_id]),
        )

        self._model_predictions = model_predictions_matched_filtered.copy()[
            model_predictions_matched_filtered[ClassificationPredictionMatchSchemaWithClassNames.img_id].isin(
                img_id_intersection
            )
        ]

        return True

    def render_view_options(self):
        if not get_state().predictions.all_classes_classifications:
            get_state().predictions.all_classes_classifications = read_class_idx(
                get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
            )

        if self.page_mode != ModelQualityPage.EXPLORER:
            get_state().predictions.selected_classes_classifications = self._render_class_filtering_component(
                get_state().predictions.all_classes_classifications
            )
            self._topbar_additional_settings()

        render_filter()

    def _topbar_additional_settings(self):
        if self.page_mode == ModelQualityPage.METRICS:
            return
        elif self.page_mode == ModelQualityPage.PERFORMANCE_BY_METRIC:
            c1, c2, c3 = st.columns([4, 4, 3])
            with c1:
                self._topbar_metric_selection_component(
                    MetricType.PREDICTION, get_state().predictions.metric_datas_classification
                )
            with c2:
                self._set_binning()
            with c3:
                self._class_decomposition()
        elif self.page_mode == ModelQualityPage.EXPLORER:
            c1, c2 = st.columns([4, 4])

            with c1:
                self._explorer_outcome_type = st.selectbox(
                    "Outcome",
                    [x for x in ClassificationOutcomeType],
                    format_func=lambda x: x.value,
                    help="Only the samples with this outcome will be shown",
                )
            with c2:
                self._topbar_metric_selection_component(
                    MetricType.PREDICTION, get_state().predictions.metric_datas_classification
                )

    def _render_magic_search_pane(self):
        querier = Querier(get_state().project_paths)

        is_disabled = not all([get_settings().ENV == Env.PROD, querier.premium_available])

        prepared_prompt = st.selectbox(
            "Some prompts to start with",
            [
                "What are the least three performing classes in terms of mean prediction",
                "What is the lowest performing classes in terms of mean prediction for small images",
                "What classes' prediction is low when the brightness is high",
                "What are the lowest performing classes in terms of mean prediction value when the area of the image is low",
            ],
            disabled=is_disabled,
        )

        magic_prompt = st.text_input(
            "🪄 What do you want to get?",
            value=prepared_prompt,
            disabled=is_disabled,
            help="Only available for premium version",
        )

        if magic_prompt != "" and not is_disabled:
            selected_ids = list(self._model_predictions[ClassificationPredictionMatchSchemaWithClassNames.identifier])
            result = querier.search_with_code_on_dataframe(TextQuery(identifiers=selected_ids, text=magic_prompt))

            if result is None:
                st.write("Server error")
                return

            if result.code is None:
                st.write("Code could not be generated for this prompt.")
                return

            st.code(result.code)
            if result.output is not None:
                st.write(result.output)
            else:
                st.write("An output could not obtained for this code")

    def is_available(self) -> bool:
        return check_model_prediction_availability(
            get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
        )

    def render_metrics(self):
        class_names = sorted(list(set(self._labels).union(self._predictions)))

        precision, recall, f1, support = get_precision_recall_f1(self._labels, self._predictions)
        accuracy = get_accuracy(self._labels, self._predictions)

        # PERFORMANCE METRICS SUMMARY

        col_acc, col_prec, col_rec, col_f1 = st.columns(4)
        col_acc.metric("Accuracy", f"{float(accuracy):.2f}")
        col_prec.metric(
            "Mean Precision",
            f"{float(precision.mean()):.2f}",
            help="Average of precision scores of all classes",
        )
        col_rec.metric("Mean Recall", f"{float(recall.mean()):.2f}", help="Average of recall scores of all classes")
        col_f1.metric("Mean F1", f"{float(f1.mean()):.2f}", help="Average of F1 scores of all classes")

        # METRIC IMPORTANCE
        self._get_metric_importance(
            self._model_predictions, list(get_state().predictions.metric_datas_classification.predictions.keys())
        )

        col1, col2 = st.columns(2)

        # CONFUSION MATRIX
        confusion_matrix = get_confusion_matrix(self._labels, self._predictions, class_names)
        col1.plotly_chart(confusion_matrix, use_container_width=True)

        # PRECISION_RECALL BARS
        pr_graph = get_precision_recall_graph(precision, recall, class_names)
        col2.plotly_chart(pr_graph, use_container_width=True)

        # Magic search on the dataframe
        self._render_magic_search_pane()

    def render_performance_by_metric(self):
        self._render_performance_by_metric_description(
            self._model_predictions, get_state().predictions.metric_datas_classification
        )
        metric_name = get_state().predictions.metric_datas_classification.selected_prediction

        classes_for_coloring = ["Average"]
        decompose_classes = get_state().predictions.decompose_classes
        if decompose_classes:
            unique_classes = set(self._model_predictions["class_name"].unique())
            classes_for_coloring += sorted(list(unique_classes))

        # Ensure same colors between plots
        chart_args = dict(
            color_params={"scale": alt.Scale(domain=classes_for_coloring)},
            bins=get_state().predictions.nbins,
            show_decomposition=decompose_classes,
        )

        try:
            precision = performance_rate_by_metric(
                self._model_predictions,
                metric_name,
                scope=PredictionMatchScope.TRUE_POSITIVES,
                **chart_args,
            )
            if precision is not None:
                st.altair_chart(precision.interactive(), use_container_width=True)
        except Exception as e:
            logger.warning(e)
            pass

    def render_explorer(self):
        filters = get_state().filtering_state.filters.copy()
        filters.prediction_filters = PredictionsFilters(
            type=MainPredictionType.CLASSIFICATION,
        )

        explorer(
            auth_token=get_auth_token(),
            project_name=get_state().project_paths.project_dir.name,
            scope="model_quality",
            api_url=get_settings().API_URL,
            filters=filters.dict(),
        )
