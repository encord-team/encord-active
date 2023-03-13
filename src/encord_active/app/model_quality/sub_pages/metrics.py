import json
from typing import Optional, cast

import streamlit as st
from loguru import logger
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator

from encord_active.app.common.state import get_state
from encord_active.lib.charts.classification_metrics import (
    get_accuracy,
    get_confusion_matrix,
    get_precision_recall_f1,
    get_precision_recall_graph,
)
from encord_active.lib.charts.metric_importance import create_metric_importance_charts
from encord_active.lib.charts.precision_recall import create_pr_chart_plotly
from encord_active.lib.constants import DOCS_URL
from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)
from encord_active.lib.model_predictions.reader import (
    ClassificationPredictionMatchSchema,
    LabelMatchSchema,
    PredictionMatchSchema,
)
from encord_active.lib.model_predictions.writer import MainPredictionType

from . import ModelQualityPage

_M_COLS = PerformanceMetricSchema


class MetricsPage(ModelQualityPage):
    title = "📈 Metrics"

    def sidebar_options(self):
        pass

    def sidebar_options_classifications(self):
        pass

    def _build_objects(
        self,
        object_model_predictions: DataFrame[PredictionMatchSchema],
        object_metrics: DataFrame[PerformanceMetricSchema],
        object_precisions: DataFrame[PrecisionRecallSchema],
    ):

        _map = object_metrics[object_metrics[_M_COLS.metric] == "mAP"]["value"].item()
        _mar = object_metrics[object_metrics[_M_COLS.metric] == "mAR"]["value"].item()
        col1, col2 = st.columns(2)
        col1.metric("mAP", f"{_map:.3f}")
        col2.metric("mAR", f"{_mar:.3f}")
        st.markdown("""---""")

        st.subheader("Metric Importance")
        with st.container():
            with st.expander("Description"):
                st.write(
                    "The following charts show the dependency between model performance and each index. "
                    "In other words, these charts answer the question of how much is model "
                    "performance affected by each index. This relationship can be decomposed into two metrics:"
                )
                st.markdown(
                    "- **Metric importance**: measures the *strength* of the dependency between and metric and model "
                    "performance. A high value means that the model performance would be strongly affected by "
                    "a change in the index. For example, a high importance in 'Brightness' implies that a change "
                    "in that quantity would strongly affect model performance. Values range from 0 (no dependency) "
                    "to 1 (perfect dependency, one can completely predict model performance simply by looking "
                    "at this index)."
                )
                st.markdown(
                    "- **Metric [correlation](https://en.wikipedia.org/wiki/Correlation)**: measures the *linearity "
                    "and direction* of the dependency between an index and model performance. "
                    "Crucially, this metric tells us whether a positive change in an index "
                    "will lead to a positive change (positive correlation) or a negative change (negative correlation) "
                    "in model performance . Values range from -1 to 1."
                )
                st.write(
                    "Finally, you can also select how many samples are included in the computation "
                    "with the slider, as well as filter by class with the dropdown in the side bar."
                )

            if (
                object_model_predictions.shape[0] > 60_000
            ):  # Computation are heavy so allow computing for only a subset.
                num_samples = st.slider(
                    "Number of samples",
                    min_value=1,
                    max_value=len(object_model_predictions),
                    step=max(1, (len(object_model_predictions) - 1) // 100),
                    value=max((len(object_model_predictions) - 1) // 2, 1),
                    help="To avoid too heavy computations, we subsample the data at random to the selected size, "
                    "computing importance values.",
                )
                if num_samples < 100:
                    st.warning(
                        "Number of samples is too low to compute reliable index importances. "
                        "We recommend using at least 100 samples.",
                    )
            else:
                num_samples = object_model_predictions.shape[0]

            metric_columns = list(get_state().predictions.metric_datas.predictions.keys())
            with st.spinner("Computing index importance..."):
                try:
                    chart = create_metric_importance_charts(
                        object_model_predictions,
                        metric_columns=metric_columns,
                        num_samples=num_samples,
                        prediction_type=MainPredictionType.OBJECT,
                    )
                    st.altair_chart(chart, use_container_width=True)
                except ValueError as e:
                    if e.args:
                        st.info(e.args[0])
                    else:
                        st.info("Failed to compute metric importance")

        st.subheader("Subset selection scores")
        with st.container():
            project_ontology = json.loads(get_state().project_paths.ontology.read_text(encoding="utf-8"))
            chart = create_pr_chart_plotly(object_metrics, object_precisions, project_ontology["objects"])
            st.plotly_chart(chart, use_container_width=True)

    def _build_classifications(
        self,
        classification_labels: list,
        classification_pred: list,
        classification_model_predictions_matched: DataFrame[ClassificationPredictionMatchSchema],
    ):
        class_names = sorted(list(set(classification_labels).union(classification_pred)))

        precision, recall, f1, support = get_precision_recall_f1(classification_labels, classification_pred)
        accuracy = get_accuracy(classification_labels, classification_pred)

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
            classification_model_predictions_matched.shape[0] > 60_000
        ):  # Computation are heavy so allow computing for only a subset.
            num_samples = st.slider(
                "Number of samples",
                min_value=1,
                max_value=len(classification_model_predictions_matched),
                step=max(1, (len(classification_model_predictions_matched) - 1) // 100),
                value=max((len(classification_model_predictions_matched) - 1) // 2, 1),
                help="To avoid too heavy computations, we subsample the data at random to the selected size, "
                "computing importance values.",
            )
            if num_samples < 100:
                st.warning(
                    "Number of samples is too low to compute reliable index importances. "
                    "We recommend using at least 100 samples.",
                )
        else:
            num_samples = classification_model_predictions_matched.shape[0]

        metric_columns = list(get_state().predictions.metric_datas_classification.predictions.keys())
        try:
            metric_importance_chart = create_metric_importance_charts(
                classification_model_predictions_matched,
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
        confusion_matrix = get_confusion_matrix(classification_labels, classification_pred, class_names)
        col1.plotly_chart(confusion_matrix, use_container_width=True)

        # PRECISION_RECALL BARS
        pr_graph = get_precision_recall_graph(precision, recall, class_names)
        col2.plotly_chart(pr_graph, use_container_width=True)

        # In order to plot ROC curve, we need confidences for the ground
        # truth label. Currently, predictions.pkl file only has confidence
        # value for the predicted class.
        # roc_graph = get_roc_curve(classification_labels, classification_pred)

    def build(
        self,
        object_predictions_exist: bool,
        classification_predictions_exist: bool,
        object_tab: DeltaGenerator,
        classification_tab: DeltaGenerator,
        object_model_predictions: Optional[DataFrame[PredictionMatchSchema]] = None,
        object_labels: Optional[DataFrame[LabelMatchSchema]] = None,
        object_metrics: Optional[DataFrame[PerformanceMetricSchema]] = None,
        object_precisions: Optional[DataFrame[PrecisionRecallSchema]] = None,
        classification_labels: Optional[list] = None,
        classification_pred: Optional[list] = None,
        classification_model_predictions_matched: Optional[DataFrame[ClassificationPredictionMatchSchema]] = None,
    ):
        with object_tab:
            if self.check_building_object_quality(
                object_predictions_exist, object_model_predictions, object_labels, object_metrics, object_precisions
            ):
                self._build_objects(object_model_predictions, object_metrics, object_precisions)

        with classification_tab:
            if self.check_building_classification_quality(
                classification_predictions_exist,
                classification_labels,
                classification_pred,
                classification_model_predictions_matched,
            ):
                self._build_classifications(
                    cast(list, classification_labels),
                    cast(list, classification_pred),
                    DataFrame[ClassificationPredictionMatchSchema](classification_model_predictions_matched),
                )
