from abc import abstractmethod
from enum import Enum
from typing import List, Union

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
    TRUE_POSITIVES = "true positives"
    FALSE_POSITIVES = "false positives"
    FALSE_NEGATIVES = "false negatives"


class PredictionTypeBuilder(Page):
    name: str

    @staticmethod
    def metric_details_description(metric_datas: MetricNames):
        metric_name = metric_datas.selected_prediction
        if not metric_name:
            return

        metric_data = metric_datas.predictions.get(metric_name)

        if not metric_data:
            metric_data = metric_datas.labels.get(metric_name)

        if metric_data:
            st.markdown(f"### The {metric_data.name[:-4]} metric")
            st.markdown(metric_data.meta.long_description)

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
    def _render_true_positives(self):
        pass

    @abstractmethod
    def _render_false_positives(self):
        pass

    @abstractmethod
    def _render_false_negatives(self):
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

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

    def _prediction_metric_in_sidebar_objects(self, page_mode: ModelQualityPage, metric_datas: MetricNames):
        """
        Note: Adding the fixed options "confidence" and "iou" works here because
        confidence is required on import and IOU is computed during prediction
        import. So the two columns are already available in the
        `st.session_state.model_predictions` data frame.
        """
        if page_mode == ModelQualityPage.FALSE_NEGATIVES:
            column_names = list(metric_datas.predictions.keys())
            metric_datas.selected_label = st.selectbox(
                "Select metric for your labels",
                column_names,
                help="The data in the main view will be sorted by the selected metric. "
                "(F) := frame scores, (O) := object scores.",
            )
        else:
            fixed_options = {"confidence": "Model Confidence"}
            column_names = list(metric_datas.predictions.keys())
            metric_datas.selected_prediction = st.selectbox(
                "Select metric for your predictions",
                column_names + list(fixed_options.keys()),
                format_func=lambda s: fixed_options.get(s, s),
                help="The data in the main view will be sorted by the selected metric. "
                "(F) := frame scores, (P) := prediction scores.",
            )

    def read_prediction_files(
        self, prediction_type: MainPredictionType
    ) -> (
        List[MetricData],
        List[MetricData],
        Union[DataFrame[PredictionSchema], DataFrame[ClassificationPredictionSchema], None],
        Union[DataFrame[LabelSchema], DataFrame[ClassificationLabelSchema], None],
    ):
        metrics_dir = get_state().project_paths.metrics
        predictions_dir = get_state().project_paths.predictions / prediction_type.value

        predictions_metric_datas = use_memo(lambda: reader.get_prediction_metric_data(predictions_dir, metrics_dir))

        label_metric_datas = use_memo(lambda: reader.get_label_metric_data(metrics_dir))
        model_predictions = use_memo(
            lambda: reader.get_model_predictions(predictions_dir, predictions_metric_datas, prediction_type)
        )
        labels = use_memo(lambda: reader.get_labels(predictions_dir, label_metric_datas, prediction_type))

        return predictions_metric_datas, label_metric_datas, model_predictions, labels

    def _get_metric_importance(
        self,
        model_predictions: Union[
            DataFrame[ClassificationPredictionMatchSchemaWithClassNames], DataFrame[PredictionMatchSchema]
        ],
        metric_columns: List[str],
        num_samples: int,
    ):
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
            elif page_mode == ModelQualityPage.TRUE_POSITIVES:
                self._render_true_positives()
            elif page_mode == ModelQualityPage.FALSE_POSITIVES:
                self._render_false_positives()
            elif page_mode == ModelQualityPage.FALSE_NEGATIVES:
                self._render_false_negatives()
