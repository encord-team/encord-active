from copy import deepcopy
from enum import Enum
from typing import List, Optional

import altair as alt
import streamlit as st
from loguru import logger
from pandera.typing import DataFrame

import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.components import sticky_header
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
from encord_active.lib.metrics.utils import MetricScope
from encord_active.lib.model_predictions.classification_metrics import (
    match_predictions_and_labels,
)
from encord_active.lib.model_predictions.filters import (
    prediction_and_label_filtering_classification,
)
from encord_active.lib.model_predictions.reader import (
    ClassificationLabelSchema,
    ClassificationPredictionMatchSchemaWithClassNames,
    ClassificationPredictionSchema,
    get_class_idx,
)
from encord_active.lib.model_predictions.writer import MainPredictionType


class ClassificationTypeBuilder(PredictionTypeBuilder):
    title = "Classification"

    class OutcomeType(str, Enum):
        TRUE_POSITIVES = "True Positive"
        FALSE_POSITIVES = "False Positive"

    def __init__(self):

        self._labels: Optional[List] = None
        self._predictions: Optional[List] = None
        self._model_predictions: Optional[DataFrame[ClassificationPredictionMatchSchemaWithClassNames]] = None

    def sidebar_options(self, *args, **kwargs):
        pass

    def _load_data(self, page_mode: ModelQualityPage) -> bool:
        predictions_metric_datas, label_metric_datas, model_predictions, labels = self._read_prediction_files(
            MainPredictionType.CLASSIFICATION
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

        with sticky_header():
            self._common_settings()
            self._topbar_additional_settings(page_mode)

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

    def _common_settings(self):
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

    def _topbar_additional_settings(self, page_mode: ModelQualityPage):
        if page_mode == ModelQualityPage.METRICS:
            return
        elif page_mode == ModelQualityPage.PERFORMANCE_BY_METRIC:
            c1, c2, c3 = st.columns([4, 4, 3])
            with c1:
                self._prediction_metric_in_sidebar_objects(
                    MetricType.PREDICTION, get_state().predictions.metric_datas_classification
                )
            with c2:
                self._set_binning()
            with c3:
                self._class_decomposition()
        elif page_mode == ModelQualityPage.EXPLORER:
            c1, c2 = st.columns([4, 4])

            with c1:
                self._explorer_outcome_type = st.selectbox(
                    "Outcome",
                    [x for x in self.OutcomeType],
                    format_func=lambda x: x.value,
                    help="Only the samples with this outcome will be shown",
                )
            with c2:
                self._prediction_metric_in_sidebar_objects(
                    MetricType.PREDICTION, get_state().predictions.metric_datas_classification
                )

            self.display_settings(MetricScope.MODEL_QUALITY)

    def is_available(self) -> bool:
        return reader.check_model_prediction_availability(
            get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
        )

    def _render_metrics(self):
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

    def _render_performance_by_metric(self):
        if self._model_predictions.shape[0] == 0:
            st.write("No predictions of the given class(es).")
            return

        metric_name = get_state().predictions.metric_datas_classification.selected_prediction
        if not metric_name:
            # This shouldn't happen with the current flow. The only way a user can do this
            # is if he/she write custom code to bypass running the metrics. In this case,
            # I think that it is fair to not give more information than this.
            st.write(
                "No metrics computed for the your model predictions. "
                "With `encord-active import predictions /path/to/predictions.pkl`, "
                "Encord Active will automatically run compute the metrics."
            )
            return

        self._description_expander(get_state().predictions.metric_datas_classification)

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
            tpr = performance_rate_by_metric(
                self._model_predictions,
                metric_name,
                scope=PredictionMatchScope.TRUE_POSITIVES,
                **chart_args,
            )
            if tpr is not None:
                st.altair_chart(tpr.interactive(), use_container_width=True)
        except Exception as e:
            logger.warning(e)
            pass

    def _render_explorer(self):
        with st.expander("Details"):
            if self._explorer_outcome_type == self.OutcomeType.TRUE_POSITIVES:
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

        metric_name = get_state().predictions.metric_datas_classification.selected_prediction
        if not metric_name:
            st.error("No prediction metric selected")
            return

        if self._explorer_outcome_type == self.OutcomeType.TRUE_POSITIVES:
            view_df = self._model_predictions[
                self._model_predictions[ClassificationPredictionMatchSchemaWithClassNames.is_true_positive] == 1.0
            ].dropna(subset=[metric_name])
        else:
            view_df = self._model_predictions[
                self._model_predictions[ClassificationPredictionMatchSchemaWithClassNames.is_true_positive] == 0.0
            ].dropna(subset=[metric_name])

        if view_df.shape[0] == 0:
            st.write(f"No {self._explorer_outcome_type}")
        else:
            histogram = get_histogram(view_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid_classifications(get_state().project_paths.data, model_predictions=view_df)

    def _render_true_positives(self):
        with st.expander("Details"):
            st.markdown(
                """### The view
These are the predictions where the model correctly predicts the true class.
                    """,
                unsafe_allow_html=True,
            )
            self._metric_details_description(get_state().predictions.metric_datas_classification)

        metric_name = get_state().predictions.metric_datas_classification.selected_prediction
        if not metric_name:
            st.error("No prediction metric selected")
            return

        tp_df = self._model_predictions[
            self._model_predictions[ClassificationPredictionMatchSchemaWithClassNames.is_true_positive] == 1.0
        ].dropna(subset=[metric_name])

        if tp_df.shape[0] == 0:
            st.write("No true positives")
        else:
            histogram = get_histogram(tp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid_classifications(get_state().project_paths.data, model_predictions=tp_df)

    def _render_false_positives(self):
        with st.expander("Details"):
            st.markdown(
                """### The view
These are the predictions where the model incorrectly predicts the positive class.
                    """,
                unsafe_allow_html=True,
            )
            self._metric_details_description(get_state().predictions.metric_datas_classification)

        metric_name = get_state().predictions.metric_datas_classification.selected_prediction
        if not metric_name:
            st.error("No prediction metric selected")
            return

        fp_df = self._model_predictions[
            self._model_predictions[ClassificationPredictionMatchSchemaWithClassNames.is_true_positive] == 0.0
        ].dropna(subset=[metric_name])

        if fp_df.shape[0] == 0:
            st.write("No false positives")
        else:
            histogram = get_histogram(fp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid_classifications(get_state().project_paths.data, model_predictions=fp_df)
