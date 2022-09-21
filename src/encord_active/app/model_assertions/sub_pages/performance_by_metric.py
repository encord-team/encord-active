import re
from collections import namedtuple
from enum import Enum
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

import encord_active.app.common.state as state

from . import ModelAssertionsPage

FLOAT_FMT = ",.4f"
PCT_FMT = ",.2f"
COUNT_FMT = ",d"
CompositeChart = namedtuple("CompositeChart", "chart xmin xmax")

sub_plot_args = dict(width=370, height=300)
zoom_view_args = dict(height=50, width=800, title="Select to Zoom")


class ChartSubject(Enum):
    TPR = "true positive rate"
    FNR = "false negative rate"


class PerformanceMetric(ModelAssertionsPage):
    title = "⚡️ Performance by Metric"

    def build_bar_chart(
        self,
        sorted_predictions: pd.DataFrame,
        metric_name: str,
        show_decomposition: bool,
        title: str,
        subject: ChartSubject,
    ) -> alt.Chart:
        str_type = "predictions" if subject == ChartSubject.TPR else "labels"
        largest_bin_count = sorted_predictions["bin"].value_counts().max()
        chart = (
            alt.Chart(sorted_predictions, title=title)
            .transform_joinaggregate(total="count(*)")
            .transform_calculate(
                pctf=f"1 / {largest_bin_count}",
                pct="100 / datum.total",
            )
            .mark_bar(align="center", opacity=0.2)
        )
        if show_decomposition:
            # Aggregate over each class
            return chart.encode(
                alt.X("bin:Q", scale=self.x_scale),
                alt.Y("sum(pctf):Q", stack="zero"),
                alt.Color("class_name:N", scale=self.class_scale, legend=alt.Legend(symbolOpacity=1)),
                tooltip=[
                    alt.Tooltip("bin", title=metric_name, format=FLOAT_FMT),
                    alt.Tooltip("count():Q", title=f"Num. {str_type}", format=COUNT_FMT),
                    alt.Tooltip("sum(pct):Q", title=f"% of total {str_type}", format=PCT_FMT),
                    alt.Tooltip("class_name:N", title="Class name"),
                ],
            )
        else:
            # Only use aggregate over all classes
            return chart.encode(
                alt.X("bin:Q", scale=self.x_scale),
                alt.Y("sum(pctf):Q", stack="zero"),
                tooltip=[
                    alt.Tooltip("bin", title=metric_name, format=FLOAT_FMT),
                    alt.Tooltip("count():Q", title=f"Num. {str_type}", format=COUNT_FMT),
                    alt.Tooltip("sum(pct):Q", title=f"% of total {str_type}", format=PCT_FMT),
                ],
            )

    def build_line_chart(
        self, bar_chart: alt.Chart, metric_name: str, show_decomposition: bool, title: str
    ) -> alt.Chart:
        legend = alt.Legend(title="class name".title())
        title_shorthand = "".join(w[0].upper() for w in title.split())

        line_chart = bar_chart.mark_line(point=True, opacity=0.5 if show_decomposition else 1.0).encode(
            alt.X("bin:Q", scale=self.x_scale),
            alt.Y("mean(indicator):Q"),
            alt.Color("average:N", legend=legend, scale=self.class_scale),
            tooltip=[
                alt.Tooltip("bin", title=metric_name, format=FLOAT_FMT),
                alt.Tooltip("mean(indicator):Q", title=title_shorthand, format=FLOAT_FMT),
                alt.Tooltip("average:N", title="Class name"),
            ],
            strokeDash=alt.value([5, 5]),
        )

        if show_decomposition:
            line_chart += line_chart.mark_line(point=True).encode(
                alt.Color("class_name:N", legend=legend, scale=self.class_scale),
                tooltip=[
                    alt.Tooltip("bin", title=metric_name, format=FLOAT_FMT),
                    alt.Tooltip("mean(indicator):Q", title=title_shorthand, format=FLOAT_FMT),
                    alt.Tooltip("class_name:N", title="Class name"),
                ],
                strokeDash=alt.value([10, 0]),
                opacity=alt.condition(self.class_selection, alt.value(1), alt.value(0.1)),
            )
            line_chart = line_chart.add_selection(self.class_selection)
        return line_chart

    def build_average_rule(self, indicator_mean: float, title: str):
        title_shorthand = "".join(w[0].upper() for w in title.split())
        return (
            alt.Chart(pd.DataFrame({"y": [indicator_mean], "average": ["Average"]}))
            .mark_rule()
            .encode(
                alt.Y("y"),
                alt.Color("average:N", scale=self.class_scale),
                strokeDash=alt.value([5, 5]),
                tooltip=[alt.Tooltip("y", title=f"Average {title_shorthand}", format=FLOAT_FMT)],
            )
        )

    def make_composite_chart(
        self, df: pd.DataFrame, title: str, metric_name: str, subject: ChartSubject
    ) -> Optional[CompositeChart]:
        # Avoid over-shooting number of bins.
        if metric_name not in df.columns:
            return None

        num_unique_values = df[metric_name].unique().shape[0]
        n_bins = min(st.session_state.get(state.PREDICTIONS_NBINS, 100), num_unique_values)

        # Avoid nans
        df = df[[metric_name, "indicator", "class_name"]].copy().dropna(subset=[metric_name])

        if df.empty:
            st.info(f"{title}: No values for the selected metric: {metric_name}")
            return None

        # Bin the data
        df["bin_info"] = pd.qcut(df[metric_name], q=n_bins, duplicates="drop")
        df["bin"] = df["bin_info"].map(lambda x: x.mid)

        if df["bin"].dropna().empty:
            st.info(f"No scores for the selected metric: {metric_name}")
            return None

        df["average"] = "Average"  # Indicator for altair charts
        show_decomposition = st.session_state[state.PREDICTIONS_DECOMPOSE_CLASSES]

        bar_chart = self.build_bar_chart(df, metric_name, show_decomposition, title=title, subject=subject)
        line_chart = self.build_line_chart(bar_chart, metric_name, show_decomposition, title=title)
        mean_rule = self.build_average_rule(df["indicator"].mean(), title=title)

        chart_composition = bar_chart + line_chart + mean_rule
        chart_composition = chart_composition.encode(
            alt.X(title=metric_name.title()), alt.Y(title=title.title())
        ).properties(
            **sub_plot_args
        )  # TODO Is there a better way to set size?
        return CompositeChart(chart_composition, df["bin"].min(), df["bin"].max())

    def sidebar_options(self):
        c1, c2, c3 = st.columns([4, 4, 3])
        with c1:
            self.prediction_metric_in_sidebar()

        with c2:
            st.number_input(
                "Number of buckets (n)",
                min_value=5,
                max_value=200,
                value=50,
                help="Choose the number of bins to discritize the prediction metric values into.",
                key=state.PREDICTIONS_NBINS,
            )
        with c3:
            st.write("")  # Make some spacing.
            st.write("")
            st.checkbox(
                "Show class decomposition",
                key=state.PREDICTIONS_DECOMPOSE_CLASSES,
                help="When checked, every plot will have a separate component for each class.",
            )

    def build(
        self,
        model_predictions: pd.DataFrame,
        labels: pd.DataFrame,
        metrics: pd.DataFrame,
        precisions: pd.DataFrame,
    ):
        st.markdown(f"# {self.title}")

        if model_predictions.shape[0] == 0:
            st.write("No predictions of the given class(es).")
        elif state.PREDICTIONS_METRIC not in st.session_state:
            # This shouldn't happen with the current flow. The only way a user can do this
            # is if he/she write custom code to bypass running the metrics. In this case,
            # I think that it is fair to not give more information than this.
            st.write(
                "No metrics computed for the your model predictions. "
                "With `encord-active import predictions /path/to/predictions.pkl`, "
                "Encord Active will automatically run compute the metrics."
            )
        else:
            with st.expander("Details", expanded=False):
                st.markdown(
                    """### The View

On this page, your model scores are displayed as a function of the metric that you selected in the top bar.
Samples are discritized into $n$ equally sized buckets and the middle point of each bucket is displayed as the x-value in the plots.
Bars indicate the number of samples in each bucket, while lines indicate the true positive and false negative rates of each bucket.


Metrics marked with (P) are metrics computed on your predictions.
Metrics marked with (F) are frame level metrics, which depends on the frame that each prediction is associated
with. In the "False Negative Rate" plot, (O) means metrics compoted on Object labels.

For metrics that are computed on predictions (P) in the "True Positive Rate" plot, the corresponding "label metrics" (O/F) computed
on your labels are used for the "False Negative Rate" plot.
""",
                    unsafe_allow_html=True,
                )
                self.metric_details_description()

            metric_name = st.session_state[state.PREDICTIONS_METRIC]

            label_metric_name = metric_name
            if metric_name[-3:] == "(P)":  # Replace the P with O:  "Metric (P)" -> "Metric (O)"
                label_metric_name = re.sub(r"(.*?\()P(\))", r"\1O\2", metric_name)

            if not label_metric_name in labels.columns:
                label_metric_name = re.sub(
                    r"(.*?\()O(\))", r"\1F\2", label_metric_name
                )  # Look for it in frame label metrics.

            classes_for_coloring = ["Average"]
            if st.session_state.get(state.PREDICTIONS_DECOMPOSE_CLASSES, False):
                unique_classes = set(model_predictions["class_name"].unique()).union(labels["class_name"].unique())
                classes_for_coloring += sorted(list(unique_classes))

            # Ensure same colors between plots
            self.class_scale = alt.Scale(
                domain=classes_for_coloring,
            )  # Used to sync colors between plots.
            self.class_selection = alt.selection_multi(fields=["class_name"])  # Used to sync selections between plots.

            # Allow zooming via a "zoom_chart"
            self.x_scale_interval = alt.selection_interval(encodings=["x"])
            self.x_scale = alt.Scale(domain=self.x_scale_interval.ref())

            # TPR
            predictions = model_predictions.rename(columns={"tps": "indicator"})
            tpr = self.make_composite_chart(predictions, "True Positive Rate", metric_name, subject=ChartSubject.TPR)
            if tpr is None:
                st.stop()

            # FNR
            fnr = self.make_composite_chart(
                labels.rename(columns={"fns": "indicator"}),
                "False Negative Rate",
                label_metric_name,
                subject=ChartSubject.FNR,
            )

            if fnr is None:  # Label metric couldn't be matched to
                zoom_chart = (
                    tpr.chart.encode(
                        alt.X("bin:Q"), alt.Y(title="TPR")  # Avoid zooming the actual zoom selection view.
                    )
                    .properties(**zoom_view_args)
                    .add_selection(self.x_scale_interval)
                )
                st.altair_chart(zoom_chart & tpr.chart.properties(width=800))
                st.stop()

            # Zoom chart - get entire range of boths tprs and fnrs
            xmin = min(tpr.xmin, fnr.xmin)
            xmax = max(tpr.xmax, fnr.xmax)
            zoom_chart = (
                tpr.chart.encode(
                    alt.X(
                        "bin:Q", scale=alt.Scale(domain=[xmin, xmax])
                    ),  # Avoid zooming the actual zoom selection view.
                    alt.Y(title="TPR"),
                )
                .properties(**zoom_view_args)
                .add_selection(self.x_scale_interval)
            )

            st.altair_chart(zoom_chart & (tpr.chart | fnr.chart))
