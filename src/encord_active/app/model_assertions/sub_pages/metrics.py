import altair as alt
import pandas as pd
import streamlit as st
from sklearn.feature_selection import mutual_info_regression

from . import ModelAssertionsPage


class MetricsPage(ModelAssertionsPage):
    title = "📈 Metrics"

    @staticmethod
    def create_pr_charts(metrics, precisions):
        _metrics = metrics[~metrics["metric"].isin({"mAR", "mAP"})].copy()

        tmp = "m" + _metrics["metric"].str.split("_", n=1, expand=True)
        tmp.columns = ["group", "_"]
        _metrics["group"] = tmp["group"]
        _metrics["average"] = "average"  # Legend title

        class_selection = alt.selection_multi(fields=["class_name"])

        class_bars = (
            alt.Chart(_metrics, title="Mean scores")
            .mark_bar()
            .encode(
                alt.X("value", title="", scale=alt.Scale(domain=[0.0, 1.0])),
                alt.Y("metric", title=""),
                alt.Color("class_name"),
                tooltip=[alt.Tooltip("metric", title="Metric"), alt.Tooltip("value", title="Value", format=",.3f")],
                opacity=alt.condition(class_selection, alt.value(1), alt.value(0.1)),
            )
            .properties(height=300)
        )
        # Average
        mean_bars = class_bars.encode(
            alt.X("mean(value):Q", title="", scale=alt.Scale(domain=[0.0, 1.0])),
            alt.Y("group:N", title=""),
            alt.Color("average:N"),
            tooltip=[
                alt.Tooltip("group:N", title="Metric"),
                alt.Tooltip("mean(value):Q", title="Value", format=",.3f"),
            ],
        )
        bar_chart = (class_bars + mean_bars).add_selection(class_selection)

        class_precisions = (
            alt.Chart(precisions, title="Precision-Recall Curve")
            .mark_line(point=True)
            .encode(
                alt.X("rc_threshold", title="Recall", scale=alt.Scale(domain=[0.0, 1.0])),
                alt.Y("precision", scale=alt.Scale(domain=[0.0, 1.0])),
                alt.Color("class_name"),
                tooltip=[
                    alt.Tooltip("class_name", title="Class"),
                    alt.Tooltip("rc_threshold", title="Recall"),
                    alt.Tooltip("precision", title="Precision", format=",.3f"),
                ],
                opacity=alt.condition(class_selection, alt.value(1.0), alt.value(0.2)),
            )
            .properties(height=300)
        )
        mean_precisions = (
            class_precisions.transform_calculate(average="'average'")
            .mark_line(point=True)
            .encode(
                alt.X("rc_threshold"),
                alt.Y("average(precision):Q"),
                alt.Color("average:N"),
                tooltip=[
                    alt.Tooltip("average:N", title="Aggregate"),
                    alt.Tooltip("rc_threshold", title="Recall"),
                    alt.Tooltip("average(precision)", title="Avg. precision", format=",.3f"),
                ],
            )
        )
        precision_chart = (class_precisions + mean_precisions).add_selection(class_selection)
        return bar_chart | precision_chart

    @staticmethod
    def create_metric_importance_charts(model_predictions, num_samples):
        if num_samples < model_predictions.shape[0]:
            _predictions = model_predictions.sample(num_samples, axis=0, random_state=42)
        else:
            _predictions = model_predictions

        num_tps = (_predictions["tps"].iloc[:num_samples] != 0).sum()
        if num_tps < 50:
            st.warning(
                f"Not enough true positives ({num_tps}) to calculate reliable metric importance. "
                "Try increasing the number of samples or lower the IoU threshold in the side bar."
            )

        scores = _predictions["iou"] * _predictions["tps"]
        metrics = _predictions[st.session_state.prediction_metric_names]

        correlations = metrics.fillna(0).corrwith(scores, axis=0).to_frame("correlation")
        correlations["index"] = correlations.index.T

        mi = pd.DataFrame.from_dict(
            {"index": metrics.columns, "importance": mutual_info_regression(metrics.fillna(0), scores)}
        )
        sorted_metrics = mi.sort_values("importance", ascending=False, inplace=False)["index"].to_list()

        mutual_info_bars = (
            alt.Chart(mi, title="Metric Importance")
            .mark_bar()
            .encode(
                alt.X("importance", title="Importance", scale=alt.Scale(domain=[0.0, 1.0])),
                alt.Y("index", title="Metric", sort=sorted_metrics),
                alt.Color("importance", scale=alt.Scale(scheme="blues")),
                tooltip=[
                    alt.Tooltip("index", title="Metric"),
                    alt.Tooltip("importance", title="Importance", format=",.3f"),
                ],
            )
            .properties(height=400)
        )

        correlation_bars = (
            alt.Chart(correlations, title="Metric Correlations")
            .mark_bar()
            .encode(
                alt.X("correlation", title="Correlation", scale=alt.Scale(domain=[-1.0, 1.0])),
                alt.Y("index", title="Metric", sort=sorted_metrics),
                alt.Color("correlation", scale=alt.Scale(scheme="redyellowgreen", align=0.0)),
                tooltip=[
                    alt.Tooltip("index", title="Metric"),
                    alt.Tooltip("correlation", title="Correlation", format=",.3f"),
                ],
            )
            .properties(height=400)
        )

        return alt.hconcat(mutual_info_bars, correlation_bars).resolve_scale(color="independent")

    def sidebar_options(self):
        pass

    def build(
        self,
        model_predictions: pd.DataFrame,
        labels: pd.DataFrame,
        metrics: pd.DataFrame,
        precisions: pd.DataFrame,
    ):
        st.markdown(f"# {self.title}")

        _map = metrics[metrics["metric"] == "mAP"]["value"].item()
        _mar = metrics[metrics["metric"] == "mAR"]["value"].item()
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

            if model_predictions.shape[0] > 60_000:  # Computation are heavy so allow computing for only a subset.
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
                        "Number of samples is too low to compute reliable index importances. "
                        "We recommend using at least 100 samples.",
                    )
            else:
                num_samples = model_predictions.shape[0]

            with st.spinner("Computing index importance..."):
                chart = self.create_metric_importance_charts(
                    model_predictions,
                    num_samples=num_samples,
                )
            st.altair_chart(chart, use_container_width=True)

        st.subheader("Subset selection scores")
        with st.container():
            chart = self.create_pr_charts(metrics, precisions)
            st.altair_chart(chart, use_container_width=True)
