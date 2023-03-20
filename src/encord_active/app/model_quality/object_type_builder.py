import json
import re
from copy import deepcopy
from typing import Optional

import altair as alt
import streamlit as st
from pandera.typing import DataFrame

import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.components import sticky_header
from encord_active.app.common.components.prediction_grid import prediction_grid
from encord_active.app.common.state import (
    MetricNames,
    PredictionsState,
    State,
    get_state,
)
from encord_active.app.common.state_hooks import use_memo
from encord_active.app.model_quality.prediction_type_builder import (
    ModelQualityPage,
    PredictionTypeBuilder,
)
from encord_active.lib.charts.histogram import get_histogram
from encord_active.lib.charts.metric_importance import create_metric_importance_charts
from encord_active.lib.charts.performance_by_metric import performance_rate_by_metric
from encord_active.lib.charts.precision_recall import create_pr_chart_plotly
from encord_active.lib.charts.scopes import PredictionMatchScope
from encord_active.lib.common.colors import Color
from encord_active.lib.metrics.utils import MetricScope
from encord_active.lib.model_predictions.filters import (
    filter_labels_for_frames_wo_predictions,
    prediction_and_label_filtering,
)
from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
    compute_mAP_and_mAR,
)
from encord_active.lib.model_predictions.reader import (
    LabelMatchSchema,
    PredictionMatchSchema,
    get_class_idx,
)
from encord_active.lib.model_predictions.writer import MainPredictionType


class ObjectTypeBuilder(PredictionTypeBuilder):
    name = "Object"
    title = "Object-level model performance"

    def __init__(self):
        self._model_predictions: Optional[DataFrame[PredictionMatchSchema]] = None
        self._labels: Optional[DataFrame[LabelMatchSchema]] = None
        self._metrics: Optional[DataFrame[PerformanceMetricSchema]] = None
        self._precisions: Optional[DataFrame[PrecisionRecallSchema]] = None

    def sidebar_options(self, *args, **kwargs):
        pass

    def description_expander(self, metric_datas: MetricNames):
        with st.expander("Details", expanded=False):
            st.markdown(
                """### The View

On this page, your model scores are displayed as a function of the metric that you selected in the top bar.
Samples are discritized into $n$ equally sized buckets and the middle point of each bucket is displayed as the 
x-value in the plots. Bars indicate the number of samples in each bucket, while lines indicate the true positive 
and false negative rates of each bucket.


Metrics marked with (P) are metrics computed on your predictions.
Metrics marked with (F) are frame level metrics, which depends on the frame that each prediction is associated
with. In the "False Negative Rate" plot, (O) means metrics computed on Object labels.

For metrics that are computed on predictions (P) in the "True Positive Rate" plot, the corresponding "label metrics" 
(O/F) computed on your labels are used for the "False Negative Rate" plot.
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
            metric_columns = list(get_state().predictions.metric_datas.labels.keys())
            get_state().predictions.metric_datas.selected_label = st.selectbox(
                "Select metric for your labels",
                metric_columns,
                help="The data in the main view will be sorted by the selected metric. "
                "(F) := frame scores, (O) := object scores.",
            )
            # self.display_settings(MetricScope.MODEL_QUALITY)
        else:
            fixed_options = {"confidence": "Model Confidence", "iou": "IOU"}
            column_names = list(get_state().predictions.metric_datas.predictions.keys())
            get_state().predictions.metric_datas.selected_prediction = st.selectbox(
                "Select metric for your predictions",
                column_names + list(fixed_options.keys()),
                format_func=lambda s: fixed_options.get(s, s),
                help="The data in the main view will be sorted by the selected metric. "
                "(F) := frame scores, (P) := prediction scores.",
            )

    def _load_data(self, page_mode: ModelQualityPage):
        metrics_dir = get_state().project_paths.metrics

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
            self._common_settings()
            self._topbar_additional_settings(page_mode)

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

        self._labels, self._metrics, self._model_predictions, self._precisions = prediction_and_label_filtering(
            get_state().predictions.selected_classes_objects,
            labels_filtered,
            metrics,
            sorted_model_predictions,
            precisions,
        )

    def is_available(self) -> bool:
        return reader.check_model_prediction_availability(
            get_state().project_paths.predictions / MainPredictionType.OBJECT.value
        )

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
            ModelQualityPage.FALSE_NEGATIVES,
        ]:
            self.display_settings(MetricScope.MODEL_QUALITY)

    def _common_settings(self):
        if not get_state().predictions.all_classes_objects:
            get_state().predictions.all_classes_objects = get_class_idx(
                get_state().project_paths.predictions / MainPredictionType.OBJECT.value
            )

        all_classes = get_state().predictions.all_classes_objects
        col1, col2, col3 = st.columns([4, 4, 3])

        with col1:
            selected_classes = st.multiselect(
                "Filter by class",
                list(all_classes.items()),
                format_func=lambda x: x[1]["name"],
                help="""
                With this selection, you can choose which classes to include in the performance metrics calculations.\n
                This acts as a filter, i.e. when nothing is selected all classes are included.
                """,
            )

        get_state().predictions.selected_classes_objects = dict(selected_classes) or deepcopy(all_classes)

        with col2:
            # IOU
            get_state().iou_threshold = st.slider(
                "Select an IOU threshold",
                min_value=0.0,
                max_value=1.0,
                value=State.iou_threshold,
                help="The mean average precision (mAP) score is based on true positives and false positives. "
                "The IOU threshold determines how closely predictions need to match labels to be considered "
                "as true positives.",
            )

        with col3:
            st.write("")
            st.write("")
            # Ignore unmatched frames
            get_state().ignore_frames_without_predictions = st.checkbox(
                "Ignore frames without predictions",
                value=State.ignore_frames_without_predictions,
                help="Scores like mAP and mAR are effected negatively if there are frames in the dataset for /"
                "which there exist no predictions. With this flag, you can ignore those.",
            )

    def _render_metrics(self):
        _map = self._metrics[self._metrics[PerformanceMetricSchema.metric] == "mAP"]["value"].item()
        _mar = self._metrics[self._metrics[PerformanceMetricSchema.metric] == "mAR"]["value"].item()
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

            if self._model_predictions.shape[0] > 60_000:  # Computation are heavy so allow computing for only a subset.
                num_samples = st.slider(
                    "Number of samples",
                    min_value=1,
                    max_value=len(self._model_predictions),
                    step=max(1, (len(self._model_predictions) - 1) // 100),
                    value=max((len(self._model_predictions) - 1) // 2, 1),
                    help="To avoid too heavy computations, we subsample the data at random to the selected size, "
                    "computing importance values.",
                )
                if num_samples < 100:
                    st.warning(
                        "Number of samples is too low to compute reliable index importances. "
                        "We recommend using at least 100 samples.",
                    )
            else:
                num_samples = self._model_predictions.shape[0]

            metric_columns = list(get_state().predictions.metric_datas.predictions.keys())
            with st.spinner("Computing index importance..."):
                try:
                    chart = create_metric_importance_charts(
                        self._model_predictions,
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
            chart = create_pr_chart_plotly(self._metrics, self._precisions, project_ontology["objects"])
            st.plotly_chart(chart, use_container_width=True)

    def _render_performance_by_metric(self):
        if self._model_predictions.shape[0] == 0:
            st.write("No predictions of the given class(es).")
            return

        metric_name = get_state().predictions.metric_datas.selected_prediction
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

        if not label_metric_name in self._labels.columns:
            label_metric_name = re.sub(
                r"(.*?\()O(\))", r"\1F\2", label_metric_name
            )  # Look for it in frame label metrics.

        classes_for_coloring = ["Average"]
        decompose_classes = get_state().predictions.decompose_classes
        if decompose_classes:
            unique_classes = set(self._model_predictions["class_name"].unique()).union(
                self._labels["class_name"].unique()
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
                self._model_predictions, metric_name, scope=PredictionMatchScope.TRUE_POSITIVES, **chart_args
            )
            if tpr is not None:
                st.altair_chart(tpr.interactive(), use_container_width=True)
        except:
            pass

        try:
            fnr = performance_rate_by_metric(
                self._labels, label_metric_name, scope=PredictionMatchScope.FALSE_NEGATIVES, **chart_args
            )
            if fnr is not None:
                st.altair_chart(fnr.interactive(), use_container_width=True)
        except:
            pass

    def _render_true_positives(self):
        metric_name = get_state().predictions.metric_datas.selected_prediction
        if not metric_name:
            st.error("No prediction metric selected")
            return

        with st.expander("Details"):
            color = Color.PURPLE
            st.markdown(
                f"""### The view
These are the predictions for which the IOU was sufficiently high and the confidence score was
the highest amongst predictions that overlap with the label.

---

**Color**:
The <span style="border: solid 3px {color.value}; padding: 2px 3px 3px 3px; border-radius: 4px; color: {color.value}; 
font-weight: bold;">{color.name.lower()}</span> boxes marks the true positive predictions.
The remaining colors correspond to the dataset labels with the colors you are used to from the label editor.
                            """,
                unsafe_allow_html=True,
            )
            self.metric_details_description(get_state().predictions.metric_datas)

        tp_df = self._model_predictions[self._model_predictions[PredictionMatchSchema.is_true_positive] == 1.0].dropna(
            subset=[metric_name]
        )
        if tp_df.shape[0] == 0:
            st.write("No true positives")
        else:
            histogram = get_histogram(tp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid(get_state().project_paths.data, model_predictions=tp_df, box_color=color)

    def _render_false_positives(self):
        metric_name = get_state().predictions.metric_datas.selected_prediction
        if not metric_name:
            st.error("No prediction metric selected")
            return

        color = Color.RED
        with st.expander("Details"):
            st.markdown(
                f"""### The view
        These are the predictions for which either of the following is true
        1. The IOU between the prediction and the best matching label was too low
        2. There was another prediction with higher model confidence which matched the label already
        3. The predicted class didn't match

        ---

        **Color**:
        The <span style="border: solid 3px {color.value}; padding: 2px 3px 3px 3px; border-radius: 4px; color: {color.value}; font-weight: bold;">{color.name.lower()}</span> boxes marks the false positive predictions.
        The remaining colors correspond to the dataset labels with the colors you are used to from the label editor.
                """,
                unsafe_allow_html=True,
            )
            self.metric_details_description(get_state().predictions.metric_datas)

        fp_df = self._model_predictions[self._model_predictions[PredictionMatchSchema.is_true_positive] == 0.0].dropna(
            subset=[metric_name]
        )
        if fp_df.shape[0] == 0:
            st.write("No false positives")
        else:
            histogram = get_histogram(fp_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid(get_state().project_paths.data, model_predictions=fp_df, box_color=color)

    def _render_false_negatives(self):
        metric_name = get_state().predictions.metric_datas.selected_label
        if not metric_name:
            st.error("Prediction label not selected")
            return

        with st.expander("Details"):
            color = Color.PURPLE
            st.markdown(
                f"""### The view
These are the labels that were not matched with any predictions.

---
**Color**:
The <span style="border: solid 3px {color.value}; padding: 2px 3px 3px 3px; border-radius: 4px; color: {color.value}; 
font-weight: bold;">{color.name.lower()}</span> boxes mark the false negatives. That is, the labels that were not 
matched to any predictions. The remaining objects are predictions, where colors correspond to their predicted class 
(identical colors to labels objects in the editor).
                """,
                unsafe_allow_html=True,
            )
            self.metric_details_description(get_state().predictions.metric_datas)
        fns_df = self._labels[self._labels[LabelMatchSchema.is_false_negative]].dropna(subset=[metric_name])
        if fns_df.shape[0] == 0:
            st.write("No false negatives")
        else:
            histogram = get_histogram(fns_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            prediction_grid(
                get_state().project_paths.data,
                labels=fns_df,
                model_predictions=self._model_predictions,
                box_color=color,
            )
