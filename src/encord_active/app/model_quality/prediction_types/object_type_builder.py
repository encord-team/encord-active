import json
import re
from copy import deepcopy
from enum import Enum
from typing import Optional, Union

import altair as alt
import streamlit as st
from loguru import logger
from pandera.typing import DataFrame

import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.components import sticky_header
from encord_active.app.common.components.prediction_grid import prediction_grid
from encord_active.app.common.state import MetricNames, State, get_state
from encord_active.app.common.state_hooks import use_memo
from encord_active.app.model_quality.prediction_type_builder import (
    MetricType,
    ModelQualityPage,
    PredictionTypeBuilder,
)
from encord_active.lib.charts.histogram import get_histogram
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
    title = "Object"

    class OutcomeType(str, Enum):
        TRUE_POSITIVES = "True Positive"
        FALSE_POSITIVES = "False Positive"
        FALSE_NEGATIVES = "False Negative"

    def __init__(self):
        self._explorer_outcome_type = self.OutcomeType.TRUE_POSITIVES
        self._model_predictions: Optional[DataFrame[PredictionMatchSchema]] = None
        self._labels: Optional[DataFrame[LabelMatchSchema]] = None
        self._metrics: Optional[DataFrame[PerformanceMetricSchema]] = None
        self._precisions: Optional[DataFrame[PrecisionRecallSchema]] = None

    def load_data(self, page_mode: ModelQualityPage) -> bool:
        predictions_dir = get_state().project_paths.predictions / MainPredictionType.OBJECT.value
        predictions_metric_datas, label_metric_datas, model_predictions, labels = self._read_prediction_files(
            MainPredictionType.OBJECT
        )

        if model_predictions is None:
            st.error("Couldn't load model predictions")
            return False

        if labels is None:
            st.error("Couldn't load labels properly")
            return False

        get_state().predictions.metric_datas = MetricNames(
            predictions={m.name: m for m in predictions_metric_datas},
            labels={m.name: m for m in label_metric_datas},
        )

        with sticky_header():
            self._common_settings()
            self._topbar_additional_settings(page_mode)

        matched_gt = use_memo(lambda: reader.get_gt_matched(predictions_dir))
        if not matched_gt:
            st.error("Couldn't match ground truths")
            return False

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

        return True

    def _common_settings(self):
        if not get_state().predictions.all_classes_objects:
            get_state().predictions.all_classes_objects = get_class_idx(
                get_state().project_paths.predictions / MainPredictionType.OBJECT.value
            )

        all_classes = get_state().predictions.all_classes_objects
        col1, col2, col3 = st.columns([4, 4, 3])

        with col1:
            selected_classes = self._render_class_filtering_component(all_classes)

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

    def _topbar_additional_settings(self, page_mode: ModelQualityPage):
        if page_mode == ModelQualityPage.METRICS:
            return
        elif page_mode == ModelQualityPage.PERFORMANCE_BY_METRIC:
            c1, c2, c3 = st.columns([4, 4, 3])
            with c1:
                self._prediction_metric_in_sidebar_objects(MetricType.PREDICTION, get_state().predictions.metric_datas)
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
                explorer_metric_type = (
                    MetricType.PREDICTION
                    if self._explorer_outcome_type
                    in [self.OutcomeType.TRUE_POSITIVES, self.OutcomeType.FALSE_POSITIVES]
                    else MetricType.LABEL
                )

                self._prediction_metric_in_sidebar_objects(explorer_metric_type, get_state().predictions.metric_datas)

            self.display_settings(MetricScope.MODEL_QUALITY)

    def _get_metric_name(self) -> Optional[str]:
        if self._explorer_outcome_type in [self.OutcomeType.TRUE_POSITIVES, self.OutcomeType.FALSE_POSITIVES]:
            return get_state().predictions.metric_datas.selected_prediction
        else:
            return get_state().predictions.metric_datas.selected_label

    def _render_explorer_details(self) -> Optional[Color]:
        color: Optional[Color] = None
        with st.expander("Details"):
            if self._explorer_outcome_type == self.OutcomeType.TRUE_POSITIVES:
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

            elif self._explorer_outcome_type == self.OutcomeType.FALSE_POSITIVES:
                color = Color.RED

                st.markdown(
                    f"""### The view
These are the predictions for which either of the following is true
1. The IOU between the prediction and the best matching label was too low
2. There was another prediction with higher model confidence which matched the label already
3. The predicted class didn't match

---

**Color**:
The <span style="border: solid 3px {color.value}; padding: 2px 3px 3px 3px; border-radius: 4px; color: {color.value}; 
font-weight: bold;">{color.name.lower()}</span> boxes marks the false positive predictions.
The remaining colors correspond to the dataset labels with the colors you are used to from the label editor.
                            """,
                    unsafe_allow_html=True,
                )

            elif self._explorer_outcome_type == self.OutcomeType.FALSE_NEGATIVES:
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
            self._metric_details_description(get_state().predictions.metric_datas)
            return color

    def _get_target_df(
        self, metric_name: str
    ) -> Union[DataFrame[PredictionMatchSchema], DataFrame[LabelMatchSchema], None]:

        if self._model_predictions is not None and self._labels is not None:
            if self._explorer_outcome_type == self.OutcomeType.TRUE_POSITIVES:
                return self._model_predictions[
                    self._model_predictions[PredictionMatchSchema.is_true_positive] == 1.0
                ].dropna(subset=[metric_name])
            elif self._explorer_outcome_type == self.OutcomeType.FALSE_POSITIVES:
                return self._model_predictions[
                    self._model_predictions[PredictionMatchSchema.is_true_positive] == 0.0
                ].dropna(subset=[metric_name])
            elif self._explorer_outcome_type == self.OutcomeType.FALSE_NEGATIVES:
                return self._labels[self._labels[LabelMatchSchema.is_false_negative]].dropna(subset=[metric_name])
            else:
                return None
        else:
            return None

    def is_available(self) -> bool:
        return reader.check_model_prediction_availability(
            get_state().project_paths.predictions / MainPredictionType.OBJECT.value
        )

    def render_metrics(self):
        _map = self._metrics[self._metrics[PerformanceMetricSchema.metric] == "mAP"]["value"].item()
        _mar = self._metrics[self._metrics[PerformanceMetricSchema.metric] == "mAR"]["value"].item()
        col1, col2 = st.columns(2)
        col1.metric("mAP", f"{_map:.3f}")
        col2.metric("mAR", f"{_mar:.3f}")

        # METRIC IMPORTANCE
        self._get_metric_importance(
            self._model_predictions, list(get_state().predictions.metric_datas.predictions.keys())
        )

        st.subheader("Subset selection scores")
        with st.container():
            project_ontology = json.loads(get_state().project_paths.ontology.read_text(encoding="utf-8"))
            chart = create_pr_chart_plotly(self._metrics, self._precisions, project_ontology["objects"])
            st.plotly_chart(chart, use_container_width=True)

    def render_performance_by_metric(self):
        self._render_performance_by_metric_description(self._model_predictions, get_state().predictions.metric_datas)
        metric_name = get_state().predictions.metric_datas.selected_prediction

        label_metric_name = metric_name
        if metric_name[-3:] == "(P)":  # Replace the P with O:  "Metric (P)" -> "Metric (O)"
            label_metric_name = re.sub(r"(.*?\()P(\))", r"\1O\2", metric_name)

        if label_metric_name not in self._labels.columns:
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
            if metric_name in self._model_predictions.columns:
                tpr = performance_rate_by_metric(
                    self._model_predictions, metric_name, scope=PredictionMatchScope.TRUE_POSITIVES, **chart_args
                )
                if tpr is not None:
                    st.altair_chart(tpr.interactive(), use_container_width=True)
            else:
                st.info(f"True Positive Rate is not available for `{metric_name}` metric")
        except Exception as e:
            logger.warning(e)
            pass

        try:
            if label_metric_name in self._labels.columns:
                fnr = performance_rate_by_metric(
                    self._labels, label_metric_name, scope=PredictionMatchScope.FALSE_NEGATIVES, **chart_args
                )
                if fnr is not None:
                    st.altair_chart(fnr.interactive(), use_container_width=True)
            else:
                st.info(f"False Negative Rate is not available for `{label_metric_name}` metric")
        except Exception as e:
            logger.warning(e)
            pass

    def render_explorer(self):
        metric_name = self._get_metric_name()
        if not metric_name:
            st.error("No metric selected")
            return

        color = self._render_explorer_details()
        if color is None:
            st.warning("An error occurred while rendering the explorer page")

        view_df = self._get_target_df(metric_name)
        if view_df is None:
            st.error(f"An error occurred during getting data according to the {metric_name} metric")
            return

        if view_df.shape[0] == 0:
            st.write(f"No {self._explorer_outcome_type}")
        else:
            histogram = get_histogram(view_df, metric_name)
            st.altair_chart(histogram, use_container_width=True)
            if self._explorer_outcome_type in [self.OutcomeType.TRUE_POSITIVES, self.OutcomeType.FALSE_POSITIVES]:
                prediction_grid(get_state().project_paths.data, model_predictions=view_df, box_color=color)
            else:
                prediction_grid(
                    get_state().project_paths.data,
                    model_predictions=self._model_predictions,
                    labels=view_df,
                    box_color=color,
                )
