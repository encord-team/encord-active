from copy import deepcopy
from typing import List, Optional

import altair as alt
import streamlit as st
from pandera.typing import DataFrame

import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.components import sticky_header
from encord_active.app.common.components.prediction_grid import (
    prediction_grid,
    prediction_grid_classifications,
)
from encord_active.app.common.state import MetricNames, PredictionsState, get_state
from encord_active.app.common.state_hooks import use_memo
from encord_active.app.model_quality.prediction_type_builder import (
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
from encord_active.lib.charts.metric_importance import create_metric_importance_charts
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
    name = "Classification"
    title = "Frame-level model performance"

    def __init__(self):

        self._labels: Optional[List] = None
        self._predictions: Optional[List] = None
        self._model_predictions_matched: Optional[DataFrame[ClassificationPredictionMatchSchemaWithClassNames]] = None

    def description_expander(self, metric_datas: MetricNames):
        with st.expander("Details", expanded=False):
            st.markdown(
                """### The View

On this page, your model scores are displayed as a function of the metric that you selected in the top bar.
Samples are discritized into $n$ equally sized buckets and the middle point of each bucket is displayed as the x-value in the plots.
Bars indicate the number of samples in each bucket, while lines indicate the true positive and false negative rates of each bucket.


Metrics marked with (P) are metrics computed on your predictions.
Metrics marked with (F) are frame level metrics, which depends on the frame that each prediction is associated
with. In the "False Negative Rate" plot, (O) means metrics computed on Object labels.

For metrics that are computed on predictions (P) in the "True Positive Rate" plot, the corresponding "label metrics" (O/F) computed
on your labels are used for the "False Negative Rate" plot.
""",
                unsafe_allow_html=True,
            )
            self.metric_details_description(metric_datas)

    def _prediction_metric_in_sidebar_objects(self, page_mode: ModelQualityPage):
        """
        Note: Adding the fixed options "confidence" and "iou" works here because
        confidence is required on import and IOU is computed during prediction
        import. So the two columns are already available in the
        `st.session_state.model_predictions` data frame.
        """
        if page_mode == ModelQualityPage.FALSE_NEGATIVES:
            column_names = list(get_state().predictions.metric_datas_classification.predictions.keys())
            get_state().predictions.metric_datas.selected_label = st.selectbox(
                "Select metric for your labels",
                column_names,
                help="The data in the main view will be sorted by the selected metric. "
                "(F) := frame scores, (O) := object scores.",
            )
        else:
            fixed_options = {"confidence": "Model Confidence"}
            column_names = list(get_state().predictions.metric_datas_classification.predictions.keys())
            get_state().predictions.metric_datas_classification.selected_prediction = st.selectbox(
                "Select metric for your predictions",
                column_names + list(fixed_options.keys()),
                format_func=lambda s: fixed_options.get(s, s),
                help="The data in the main view will be sorted by the selected metric. "
                "(F) := frame scores, (P) := prediction scores.",
            )

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
                self._prediction_metric_in_sidebar_objects(page_mode)

            with c2:
                get_state().predictions.nbins = int(
                    st.number_input(
                        "Number of buckets (n)",
                        min_value=5,
                        max_value=200,
                        value=PredictionsState.nbins,
                        help="Choose the number of bins to discritize the prediction metric values into.",
                    )
                )
            with c3:
                st.write("")  # Make some spacing.
                st.write("")
                get_state().predictions.decompose_classes = st.checkbox(
                    "Show class decomposition",
                    value=PredictionsState.decompose_classes,
                    help="When checked, every plot will have a separate component for each class.",
                )
        elif page_mode in [
            ModelQualityPage.TRUE_POSITIVES,
            ModelQualityPage.FALSE_POSITIVES,
            ModelQualityPage.FALSE_NEGATIVES,
        ]:
            self._prediction_metric_in_sidebar_objects(page_mode)

        if page_mode in [
            ModelQualityPage.TRUE_POSITIVES,
            ModelQualityPage.FALSE_POSITIVES,
        ]:
            self.display_settings(MetricScope.MODEL_QUALITY)

    def _load_data(self, page_mode: ModelQualityPage):
        metrics_dir = get_state().project_paths.metrics

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
            self._common_settings()
            self._topbar_additional_settings(page_mode)

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

        self._labels, self._predictions = (
            list(labels_filtered_intersection[ClassificationLabelSchema.class_id]),
            list(predictions_filtered_intersection[ClassificationPredictionSchema.class_id]),
        )

        self._model_predictions_matched = model_predictions_matched_filtered.copy()[
            model_predictions_matched_filtered[ClassificationPredictionMatchSchemaWithClassNames.img_id].isin(
                img_id_intersection
            )
        ]

    def sidebar_options(self, *args, **kwargs):
        pass

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
        if (
            self._model_predictions_matched.shape[0] > 60_000
        ):  # Computation are heavy so allow computing for only a subset.
            num_samples = st.slider(
                "Number of samples",
                min_value=1,
                max_value=len(self._model_predictions_matched),
                step=max(1, (len(self._model_predictions_matched) - 1) // 100),
                value=max((len(self._model_predictions_matched) - 1) // 2, 1),
                help="To avoid too heavy computations, we subsample the data at random to the selected size, "
                "computing importance values.",
            )
            if num_samples < 100:
                st.warning(
                    "Number of samples is too low to compute reliable index importances. "
                    "We recommend using at least 100 samples.",
                )
        else:
            num_samples = self._model_predictions_matched.shape[0]

        metric_columns = list(get_state().predictions.metric_datas_classification.predictions.keys())
        try:
            metric_importance_chart = create_metric_importance_charts(
                self._model_predictions_matched,
                metric_columns=metric_columns,
                num_samples=num_samples,
                prediction_type=MainPredictionType.CLASSIFICATION,
            )
            st.altair_chart(metric_importance_chart, use_container_width=True)
        except ValueError as e:
            if e.args:
                st.info(e.args[0])
            else:
                st.info("Failed to compute metric importance")

        col1, col2 = st.columns(2)

        # CONFUSION MATRIX
        confusion_matrix = get_confusion_matrix(self._labels, self._predictions, class_names)
        col1.plotly_chart(confusion_matrix, use_container_width=True)

        # PRECISION_RECALL BARS
        pr_graph = get_precision_recall_graph(precision, recall, class_names)
        col2.plotly_chart(pr_graph, use_container_width=True)

    def _render_performance_by_metric(self):
        if self._model_predictions_matched.shape[0] == 0:
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

        self.description_expander(get_state().predictions.metric_datas_classification)

        classes_for_coloring = ["Average"]
        decompose_classes = get_state().predictions.decompose_classes
        if decompose_classes:
            unique_classes = set(self._model_predictions_matched["class_name"].unique())
            classes_for_coloring += sorted(list(unique_classes))

        # Ensure same colors between plots
        chart_args = dict(
            color_params={"scale": alt.Scale(domain=classes_for_coloring)},
            bins=get_state().predictions.nbins,
            show_decomposition=decompose_classes,
        )

        try:
            tpr = performance_rate_by_metric(
                self._model_predictions_matched,
                metric_name,
                scope=PredictionMatchScope.TRUE_POSITIVES,
                **chart_args,
            )
            if tpr is not None:
                st.altair_chart(tpr.interactive(), use_container_width=True)
        except:
            pass

    def _render_true_positives(self):
        with st.expander("Details"):
            st.markdown(
                """### The view
These are the predictions where the model correctly predicts the true class.
                    """,
                unsafe_allow_html=True,
            )
            self.metric_details_description(get_state().predictions.metric_datas_classification)

        metric_name = get_state().predictions.metric_datas_classification.selected_prediction
        if not metric_name:
            st.error("No prediction metric selected")
            return

        tp_df = self._model_predictions_matched[
            self._model_predictions_matched[ClassificationPredictionMatchSchemaWithClassNames.is_true_positive] == 1.0
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
            self.metric_details_description(get_state().predictions.metric_datas_classification)

        metric_name = get_state().predictions.metric_datas_classification.selected_prediction
        if not metric_name:
            st.error("No prediction metric selected")
            return

        fp_df = self._model_predictions_matched[
            self._model_predictions_matched[ClassificationPredictionMatchSchemaWithClassNames.is_true_positive] == 0.0
        ].dropna(subset=[metric_name])

        if fp_df.shape[0] == 0:
            st.write("No false positives")
        else:
            histogram = get_histogram(fp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid_classifications(get_state().project_paths.data, model_predictions=fp_df)

    def _render_false_negatives(self):
        st.markdown(
            "## False Negatives view for the classification predictions is not available\n"
            "Please use **Filter by class** field in True Positives page to inspect different classes."
        )
