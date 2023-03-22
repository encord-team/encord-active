from abc import abstractmethod
from enum import Enum
from typing import List, Tuple, Union

import streamlit as st
from pandera.typing import DataFrame

import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.page import Page
from encord_active.app.common.state import MetricNames, PredictionsState, get_state
from encord_active.app.common.state_hooks import use_memo
from encord_active.lib.charts.metric_importance import create_metric_importance_charts
from encord_active.lib.metrics.utils import MetricData
from encord_active.lib.model_predictions.reader import (
    ClassificationLabelSchema,
    ClassificationPredictionMatchSchemaWithClassNames,
    ClassificationPredictionSchema,
    LabelSchema,
    PredictionMatchSchema,
    PredictionSchema,
)
from encord_active.lib.model_predictions.writer import MainPredictionType


class ModelQualityPage(str, Enum):
    METRICS = "metrics"
    PERFORMANCE_BY_METRIC = "performance by metric"
    EXPLORER = "explorer"


class MetricType(str, Enum):
    PREDICTION = "prediction"
    LABEL = "label"


class PredictionTypeBuilder(Page):
    name: str

    def _description_expander(self, metric_datas: MetricNames):
        with st.expander("Details", expanded=False):
            st.markdown(
                """### The View

On this page, your model scores are displayed as a function of the metric that you selected in the top bar.
Samples are discritized into $n$ equally sized buckets and the middle point of each bucket is displayed as the x-value 
in the plots. Bars indicate the number of samples in each bucket, while lines indicate the true positive and false 
negative rates of each bucket.

Metrics marked with (P) are metrics computed on your predictions.
Metrics marked with (F) are frame level metrics, which depends on the frame that each prediction is associated
with. In the "False Negative Rate" plot, (O) means metrics computed on Object labels.

For metrics that are computed on predictions (P) in the "True Positive Rate" plot, the corresponding "label metrics" 
(O/F) computed on your labels are used for the "False Negative Rate" plot.
""",
                unsafe_allow_html=True,
            )
            self._metric_details_description(metric_datas)

    @staticmethod
    def _metric_details_description(metric_datas: MetricNames):
        metric_name = metric_datas.selected_prediction
        if not metric_name:
            return

        metric_data = metric_datas.predictions.get(metric_name)

        if not metric_data:
            metric_data = metric_datas.labels.get(metric_name)

        if metric_data:
            st.markdown(f"### The {metric_data.name[:-4]} metric")
            st.markdown(metric_data.meta.long_description)

    def _set_binning(self):
        get_state().predictions.nbins = int(
            st.number_input(
                "Number of buckets (n)",
                min_value=5,
                max_value=200,
                value=PredictionsState.nbins,
                help="Choose the number of bins to discritize the prediction metric values into.",
            )
        )

    def _set_sampling_for_metric_importance(
        self,
        model_predictions: Union[
            DataFrame[ClassificationPredictionMatchSchemaWithClassNames], DataFrame[PredictionMatchSchema]
        ],
    ) -> int:
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
                "Number of samples is too low to compute reliable index importance. "
                "We recommend using at least 100 samples.",
            )

        return num_samples

    def _class_decomposition(self):
        st.write("")  # Make some spacing.
        st.write("")
        get_state().predictions.decompose_classes = st.checkbox(
            "Show class decomposition",
            value=PredictionsState.decompose_classes,
            help="When checked, every plot will have a separate component for each class.",
        )

    def _prediction_metric_in_sidebar_objects(self, metric_type: MetricType, metric_datas: MetricNames):
        """
        Note: Adding the fixed options "confidence" and "iou" works here because
        confidence is required on import and IOU is computed during prediction
        import. So the two columns are already available in the
        `st.session_state.model_predictions` data frame.
        """
        if metric_type == MetricType.LABEL:
            column_names = list(metric_datas.predictions.keys())
            metric_datas.selected_label = st.selectbox(
                "Select metric for your labels",
                column_names,
                help="The data in the main view will be sorted by the selected metric. "
                "(F) := frame scores, (O) := object scores.",
            )
        elif metric_type == MetricType.PREDICTION:
            fixed_options = {"confidence": "Model Confidence"}
            column_names = list(metric_datas.predictions.keys())
            metric_datas.selected_prediction = st.selectbox(
                "Select metric for your predictions",
                column_names + list(fixed_options.keys()),
                format_func=lambda s: fixed_options.get(s, s),
                help="The data in the main view will be sorted by the selected metric. "
                "(F) := frame scores, (P) := prediction scores.",
            )

    def _read_prediction_files(
        self, prediction_type: MainPredictionType
    ) -> Tuple[
        List[MetricData],
        List[MetricData],
        Union[DataFrame[PredictionSchema], DataFrame[ClassificationPredictionSchema], None],
        Union[DataFrame[LabelSchema], DataFrame[ClassificationLabelSchema], None],
    ]:
        metrics_dir = get_state().project_paths.metrics
        predictions_dir = get_state().project_paths.predictions / prediction_type.value

        predictions_metric_datas = use_memo(lambda: reader.get_prediction_metric_data(predictions_dir, metrics_dir))

        label_metric_datas = use_memo(lambda: reader.get_label_metric_data(metrics_dir))
        model_predictions = use_memo(
            lambda: reader.get_model_predictions(predictions_dir, predictions_metric_datas, prediction_type)
        )
        labels = use_memo(lambda: reader.get_labels(predictions_dir, label_metric_datas, prediction_type))

        return predictions_metric_datas, label_metric_datas, model_predictions, labels

    def _render_performance_by_metric_description(
        self,
        model_predictions: Union[
            DataFrame[ClassificationPredictionMatchSchemaWithClassNames], DataFrame[PredictionMatchSchema]
        ],
        metric_datas: MetricNames,
    ):
        if model_predictions.shape[0] == 0:
            st.write("No predictions of the given class(es).")
            return

        metric_name = metric_datas.selected_prediction
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

        self._description_expander(metric_datas)

    def _get_metric_importance(
        self,
        model_predictions: Union[
            DataFrame[ClassificationPredictionMatchSchemaWithClassNames], DataFrame[PredictionMatchSchema]
        ],
        metric_columns: List[str],
    ):
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

            if model_predictions.shape[0] > 60_000:  # Computation are heavy so allow computing for only a subset.
                num_samples = self._set_sampling_for_metric_importance(model_predictions)
            else:
                num_samples = model_predictions.shape[0]

            with st.spinner("Computing index importance..."):
                try:
                    metric_importance_chart = create_metric_importance_charts(
                        model_predictions,
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

    def build(self, page_mode: ModelQualityPage):
        if self._load_data(page_mode):
            if page_mode == ModelQualityPage.METRICS:
                self._render_metrics()
            elif page_mode == ModelQualityPage.PERFORMANCE_BY_METRIC:
                self._render_performance_by_metric()
            elif page_mode == ModelQualityPage.EXPLORER:
                self._render_explorer()

    @abstractmethod
    def _load_data(self, page_mode: ModelQualityPage) -> bool:
        pass

    @abstractmethod
    def _render_metrics(self):
        pass

    @abstractmethod
    def _render_performance_by_metric(self):
        pass

    @abstractmethod
    def _render_explorer(self):
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass
