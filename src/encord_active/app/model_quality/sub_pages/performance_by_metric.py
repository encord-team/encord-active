import re
from typing import Optional, cast

import altair as alt
import streamlit as st
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator

import encord_active.app.common.state as state
from encord_active.app.common.state import MetricNames, PredictionsState, get_state
from encord_active.lib.charts.performance_by_metric import performance_rate_by_metric
from encord_active.lib.charts.scopes import PredictionMatchScope
from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)
from encord_active.lib.model_predictions.reader import (
    ClassificationPredictionMatchSchemaWithClassNames,
    LabelMatchSchema,
    PredictionMatchSchema,
)

from . import ModelQualityPage

FLOAT_FMT = ",.4f"
PCT_FMT = ",.2f"
COUNT_FMT = ",d"


CHART_TITLES = {
    PredictionMatchScope.TRUE_POSITIVES: "True Positive Rate",
    PredictionMatchScope.FALSE_POSITIVES: "False Positive Rate",
    PredictionMatchScope.FALSE_NEGATIVES: "False Negative Rate",
}


class PerformanceMetric(ModelQualityPage):
    title = "⚡️ Performance by Metric"

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

    def sidebar_options(self):
        c1, c2, c3 = st.columns([4, 4, 3])
        with c1:
            self.prediction_metric_in_sidebar_objects()

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

    def sidebar_options_classifications(self):
        self.prediction_metric_in_sidebar_classifications()

    def _build_objects(
        self,
        object_model_predictions: DataFrame[PredictionMatchSchema],
        object_labels: DataFrame[LabelMatchSchema],
    ):

        if object_model_predictions.shape[0] == 0:
            st.write("No predictions of the given class(es).")
            return

        metric_name = state.get_state().predictions.metric_datas.selected_prediction
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

        self.description_expander(get_state().predictions.metric_datas)

        label_metric_name = metric_name
        if metric_name[-3:] == "(P)":  # Replace the P with O:  "Metric (P)" -> "Metric (O)"
            label_metric_name = re.sub(r"(.*?\()P(\))", r"\1O\2", metric_name)

        if not label_metric_name in object_labels.columns:
            label_metric_name = re.sub(
                r"(.*?\()O(\))", r"\1F\2", label_metric_name
            )  # Look for it in frame label metrics.

        classes_for_coloring = ["Average"]
        decompose_classes = get_state().predictions.decompose_classes
        if decompose_classes:
            unique_classes = set(object_model_predictions["class_name"].unique()).union(
                object_labels["class_name"].unique()
            )
            classes_for_coloring += sorted(list(unique_classes))

        # Ensure same colors between plots
        chart_args = dict(
            color_params={"scale": alt.Scale(domain=classes_for_coloring)},
            bins=get_state().predictions.nbins,
            show_decomposition=decompose_classes,
        )

        try:
            tpr = performance_rate_by_metric(
                object_model_predictions, metric_name, scope=PredictionMatchScope.TRUE_POSITIVES, **chart_args
            )
            if tpr is not None:
                st.altair_chart(tpr.interactive(), use_container_width=True)
        except:
            pass

        try:
            fnr = performance_rate_by_metric(
                object_labels, label_metric_name, scope=PredictionMatchScope.FALSE_NEGATIVES, **chart_args
            )
            if fnr is not None:
                st.altair_chart(fnr.interactive(), use_container_width=True)
        except:
            pass

    def _build_classifications(
        self,
        classification_model_predictions_matched: DataFrame[ClassificationPredictionMatchSchemaWithClassNames],
    ):
        if classification_model_predictions_matched.shape[0] == 0:
            st.write("No predictions of the given class(es).")
            return

        metric_name = state.get_state().predictions.metric_datas_classification.selected_prediction
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
            unique_classes = set(classification_model_predictions_matched["class_name"].unique())
            classes_for_coloring += sorted(list(unique_classes))

        # Ensure same colors between plots
        chart_args = dict(
            color_params={"scale": alt.Scale(domain=classes_for_coloring)},
            bins=get_state().predictions.nbins,
            show_decomposition=decompose_classes,
        )

        try:
            tpr = performance_rate_by_metric(
                classification_model_predictions_matched,
                metric_name,
                scope=PredictionMatchScope.TRUE_POSITIVES,
                **chart_args,
            )
            if tpr is not None:
                st.altair_chart(tpr.interactive(), use_container_width=True)
        except:
            pass

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
        classification_model_predictions_matched: Optional[
            DataFrame[ClassificationPredictionMatchSchemaWithClassNames]
        ] = None,
    ):

        with object_tab:
            if self.check_building_object_quality(
                object_predictions_exist, object_model_predictions, object_labels, object_metrics, object_precisions
            ):
                self._build_objects(
                    cast(DataFrame[PredictionMatchSchema], object_model_predictions),
                    cast(DataFrame[LabelMatchSchema], object_labels),
                )

        with classification_tab:
            if self.check_building_classification_quality(
                classification_predictions_exist,
                classification_labels,
                classification_pred,
                classification_model_predictions_matched,
            ):
                self._build_classifications(
                    cast(
                        DataFrame[ClassificationPredictionMatchSchemaWithClassNames],
                        classification_model_predictions_matched,
                    )
                )
