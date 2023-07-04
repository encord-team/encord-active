from enum import Enum
from typing import List, Optional, cast

import altair as alt
import streamlit as st
from loguru import logger
from pandera.typing import DataFrame

import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.components.interactive_plots import render_plotly_events
from encord_active.app.common.components.prediction_grid import (
    prediction_grid_classifications,
)
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
from encord_active.lib.charts.histogram import get_histogram
from encord_active.lib.charts.performance_by_metric import performance_rate_by_metric
from encord_active.lib.charts.scopes import PredictionMatchScope
from encord_active.lib.embeddings.dimensionality_reduction import get_2d_embedding_data
from encord_active.lib.embeddings.types import Embedding2DSchema, Embedding2DScoreSchema
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.metrics.utils import MetricSchema
from encord_active.lib.model_predictions.classification_metrics import (
    match_predictions_and_labels,
)
from encord_active.lib.model_predictions.filters import (
    prediction_and_label_filtering_classification,
)
from encord_active.lib.model_predictions.reader import (
    ClassificationLabelSchema,
    ClassificationPredictionMatchSchema,
    ClassificationPredictionMatchSchemaWithClassNames,
    ClassificationPredictionSchema,
    get_class_idx,
)
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.premium.model import TextQuery
from encord_active.lib.premium.querier import Querier
from encord_active.server.settings import Env, get_settings


class ClassificationTypeBuilder(PredictionTypeBuilder):
    title = "Classification"

    class OutcomeType(str, Enum):
        CORRECT_CLASSIFICATIONS = "Correct Classifications"
        MISCLASSIFICATIONS = "Misclassifications"

    def __init__(self):
        self._explorer_outcome_type = self.OutcomeType.CORRECT_CLASSIFICATIONS
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
            get_state().predictions.all_classes_classifications = get_class_idx(
                get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
            )

        get_state().predictions.selected_classes_classifications = self._render_class_filtering_component(
            get_state().predictions.all_classes_classifications
        )
        self._topbar_additional_settings()

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
                    [x for x in self.OutcomeType],
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
        return reader.check_model_prediction_availability(
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
        with st.expander("Details"):
            if self._explorer_outcome_type == self.OutcomeType.CORRECT_CLASSIFICATIONS:
                view_text = "These are the predictions where the model correctly predicts the true class."
            else:
                view_text = "These are the predictions where the model incorrectly predicts the positive class."
            st.markdown(
                f"""### The view
{view_text}
                    """,
                unsafe_allow_html=True,
            )

            self._metric_details_description(get_state().predictions.metric_datas_classification)

        if EmbeddingType.IMAGE not in get_state().reduced_embeddings:
            get_state().reduced_embeddings[EmbeddingType.IMAGE] = get_2d_embedding_data(
                get_state().project_paths, EmbeddingType.IMAGE
            )

        metric_name = get_state().predictions.metric_datas_classification.selected_prediction
        if not metric_name:
            st.error("No prediction metric selected")
            return

        filtered_merged_metrics = get_state().filtering_state.merged_metrics
        value = 1.0 if self._explorer_outcome_type == self.OutcomeType.CORRECT_CLASSIFICATIONS else 0.0
        view_df = self._model_predictions[
            self._model_predictions[ClassificationPredictionMatchSchemaWithClassNames.is_true_positive] == value
        ].dropna(subset=[metric_name])

        lr_du = filtered_merged_metrics.index.str.split("_", n=2).str[0:2].str.join("_")
        view_df["data_row_id"] = view_df.identifier.str.split("_", n=2).str[0:2].str.join("_")
        view_df = view_df[view_df.data_row_id.isin(lr_du)]

        if view_df.shape[0] == 0:
            st.write("There are no predictions for the given class(es).")
            return

        if get_state().reduced_embeddings[EmbeddingType.IMAGE] is None:
            st.info("There is no 2D embedding file to display.")
        else:
            current_reduced_embedding = get_state().reduced_embeddings[EmbeddingType.IMAGE]
            current_reduced_embedding["data_row_id"] = (
                current_reduced_embedding.identifier.str.split("_", n=2).str[0:2].str.join("_")
            )

            reduced_embedding_filtered = current_reduced_embedding[current_reduced_embedding.data_row_id.isin(lr_du)]

            predictions_matched = self._model_predictions[
                [ClassificationPredictionMatchSchema.identifier, ClassificationPredictionMatchSchema.is_true_positive]
            ]

            reduced_embedding_filtered = reduced_embedding_filtered.merge(
                predictions_matched, on=ClassificationPredictionMatchSchema.identifier
            )

            reduced_embedding_filtered.drop(columns=[Embedding2DSchema.label], inplace=True)
            reduced_embedding_filtered.rename(
                columns={ClassificationPredictionMatchSchema.is_true_positive: Embedding2DSchema.label}, inplace=True
            )

            reduced_embedding_filtered[Embedding2DSchema.label] = reduced_embedding_filtered[
                Embedding2DSchema.label
            ].apply(lambda x: "True prediction" if x == 1.0 else "False prediction")
            reduced_embedding_filtered[Embedding2DScoreSchema.score] = None

            selected_rows = render_plotly_events(reduced_embedding_filtered)

            if selected_rows is not None:
                view_df = view_df[view_df[MetricSchema.identifier].isin(selected_rows[Embedding2DSchema.identifier])]

        view_df.drop("data_row_id", axis=1, inplace=True)

        if view_df.shape[0] == 0:
            st.write(f"No {self._explorer_outcome_type}")
        else:
            histogram = get_histogram(view_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid_classifications(get_state().project_paths, model_predictions=view_df)
