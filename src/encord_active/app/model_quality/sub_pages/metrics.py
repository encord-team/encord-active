import streamlit as st
from pandera.typing import DataFrame

from encord_active.app.common.state import get_state
from encord_active.lib.charts.metric_importance import create_metric_importance_charts
from encord_active.lib.charts.precision_recall import create_pr_charts
from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)
from encord_active.lib.model_predictions.reader import (
    LabelMatchSchema,
    PredictionMatchSchema,
)

from . import ModelQualityPage

_M_COLS = PerformanceMetricSchema


class MetricsPage(ModelQualityPage):
    title = "📈 Metrics"

    def sidebar_options(self):
        pass

    def build(
        self,
        model_predictions: DataFrame[PredictionMatchSchema],
        labels: DataFrame[LabelMatchSchema],
        metrics: DataFrame[PerformanceMetricSchema],
        precisions: DataFrame[PrecisionRecallSchema],
    ):
        st.markdown(f"# {self.title}")

        _map = metrics[metrics[_M_COLS.metric] == "mAP"]["value"].item()
        _mar = metrics[metrics[_M_COLS.metric] == "mAR"]["value"].item()
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

            metric_columns = list(get_state().predictions.metric_datas.predictions.keys())
            with st.spinner("Computing index importance..."):
                try:
                    chart = create_metric_importance_charts(
                        model_predictions,
                        metric_columns=metric_columns,
                        num_samples=num_samples,
                    )
                    st.altair_chart(chart, use_container_width=True)
                except ValueError as e:
                    if e.args:
                        st.info(e.args[0])
                    else:
                        st.info("Failed to compute metric importance")

        st.subheader("Subset selection scores")
        with st.container():
            chart = create_pr_charts(metrics, precisions)
            st.altair_chart(chart, use_container_width=True)
