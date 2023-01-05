from typing import Optional

import altair as alt
import altair.vegalite.v4.api as alt_api
import pandas as pd
import pandera as pa
import pandera.dtypes as dtypes
from pandera.typing import DataFrame, Series

from encord_active.lib.charts.scopes import PredictionMatchScope
from encord_active.lib.model_predictions.reader import (
    LabelMatchSchema,
    PredictionMatchSchema,
)

FLOAT_FMT = ",.4f"
PCT_FMT = ",.2f"
COUNT_FMT = ",d"


CHART_TITLES = {
    PredictionMatchScope.TRUE_POSITIVES: "True Positive Rate",
    PredictionMatchScope.FALSE_POSITIVES: "False Positive Rate",
    PredictionMatchScope.FALSE_NEGATIVES: "False Negative Rate",
}


class BinSchema(pa.SchemaModel):
    metric_value: Series[float] = pa.Field()
    class_name: Series[str] = pa.Field()
    indicator: Series[float] = pa.Field()
    bin_info: Series[dtypes.Category] = pa.Field()
    bin: Series[float] = pa.Field(coerce=True)
    average: Series[str] = pa.Field()


def bin_data(df: pd.DataFrame, metric_name: str, scope: PredictionMatchScope, bins: int = 100) -> DataFrame[BinSchema]:
    # Data validation
    indicator_column = (
        PredictionMatchSchema.is_true_positive
        if scope == PredictionMatchScope.TRUE_POSITIVES
        else LabelMatchSchema.is_false_negative
    )

    schema = pa.DataFrameSchema(
        {
            metric_name: pa.Column(
                float, nullable=True, coerce=True, checks=[pa.Check(lambda x: x.count() > 0, element_wise=False)]
            ),
            "class_name": pa.Column(str),
            indicator_column: pa.Column(float, coerce=True),
        }
    )
    df = schema.validate(df)

    df = df[[metric_name, indicator_column, "class_name"]].copy().dropna(subset=[metric_name])
    df.rename(columns={metric_name: BinSchema.metric_value, indicator_column: BinSchema.indicator}, inplace=True)

    # Avoid over-shooting number of bins.
    num_unique_values = df[BinSchema.metric_value].unique().shape[0]
    n_bins = min(bins, num_unique_values)

    # Bin the data
    df["bin_info"] = pd.qcut(df[BinSchema.metric_value], q=n_bins, duplicates="drop")
    df["bin"] = df["bin_info"].map(lambda x: x.mid)
    df["average"] = "Average"  # Indicator for altair charts
    return df.pipe(DataFrame[BinSchema])


def bin_bar_chart(
    binned_df: DataFrame[BinSchema],
    metric_name: str,
    scope: PredictionMatchScope,
    show_decomposition: bool = False,
    color_params: Optional[dict] = None,
) -> alt_api.Chart:
    str_type = "predictions" if scope == PredictionMatchScope.TRUE_POSITIVES else "labels"
    largest_bin_count = binned_df["bin"].value_counts().max()
    color_params = color_params or {}
    chart = (
        alt.Chart(binned_df)
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
            alt.X("bin:Q"),
            alt.Y("sum(pctf):Q", stack="zero"),
            alt.Color(f"{BinSchema.class_name}:N", legend=alt.Legend(symbolOpacity=1), **color_params),
            tooltip=[
                alt.Tooltip(BinSchema.bin, title=metric_name, format=FLOAT_FMT),
                alt.Tooltip("count():Q", title=f"Num. {str_type}", format=COUNT_FMT),
                alt.Tooltip("sum(pct):Q", title=f"% of total {str_type}", format=PCT_FMT),
                alt.Tooltip(f"{BinSchema.class_name}:N", title="Class name"),
            ],
        )
    else:
        # Only use aggregate over all classes
        return chart.encode(
            alt.X(f"{BinSchema.bin}:Q"),
            alt.Y("sum(pctf):Q", stack="zero"),
            tooltip=[
                alt.Tooltip(BinSchema.bin, title=metric_name, format=FLOAT_FMT),
                alt.Tooltip("count():Q", title=f"Num. {str_type}", format=COUNT_FMT),
                alt.Tooltip("sum(pct):Q", title=f"% of total {str_type}", format=PCT_FMT),
            ],
        )


def performance_rate_line_chart(
    bar_chart: alt.Chart,
    metric_name: str,
    scope: PredictionMatchScope,
    show_decomposition: bool = False,
    color_params: Optional[dict] = None,
) -> alt_api.Chart:
    legend = alt.Legend(title="class name".title())
    title_shorthand = "".join(w[0].upper() for w in CHART_TITLES[scope].split())
    color_params = color_params or {}

    line_chart = bar_chart.mark_line(point=True, opacity=0.5 if show_decomposition else 1.0).encode(
        alt.X(f"{BinSchema.bin}:Q"),
        alt.Y(f"mean({BinSchema.indicator}):Q"),
        alt.Color(f"{BinSchema.average}:N", legend=legend, **color_params),
        tooltip=[
            alt.Tooltip("bin", title=metric_name, format=FLOAT_FMT),
            alt.Tooltip(f"mean({BinSchema.indicator}):Q", title=title_shorthand, format=FLOAT_FMT),
            alt.Tooltip(f"{BinSchema.average}:N", title="Class name"),
        ],
        strokeDash=alt.value([5, 5]),
    )

    if show_decomposition:
        line_chart += line_chart.mark_line(point=True).encode(
            alt.Color(f"{BinSchema.class_name}:N", legend=legend, **color_params),
            tooltip=[
                alt.Tooltip("bin", title=metric_name, format=FLOAT_FMT),
                alt.Tooltip(f"mean({BinSchema.indicator}):Q", title=title_shorthand, format=FLOAT_FMT),
                alt.Tooltip(f"{BinSchema.class_name}:N", title="Class name"),
            ],
            strokeDash=alt.value([10, 0]),
        )
    return line_chart


def performance_average_rule(
    indicator_mean: float, scope: PredictionMatchScope, color_params: Optional[dict] = None
) -> alt_api.Chart:
    title = CHART_TITLES[scope]
    title_shorthand = "".join(w[0].upper() for w in title.split())
    color_params = color_params or {}
    return (
        alt.Chart(pd.DataFrame({"y": [indicator_mean], "average": ["Average"]}))
        .mark_rule()
        .encode(
            alt.Y("y"),
            alt.Color("average:N", **color_params),
            strokeDash=alt.value([5, 5]),
            tooltip=[alt.Tooltip("y", title=f"Average {title_shorthand}", format=FLOAT_FMT)],
        )
    )


def performance_rate_by_metric(
    df: pd.DataFrame,
    metric_name: str,
    scope: PredictionMatchScope,
    bins: int = 100,
    show_decomposition: bool = False,
    color_params: Optional[dict] = None,
) -> Optional[alt_api.LayerChart]:
    binned_df = bin_data(df, metric_name, scope, bins=bins)
    if binned_df.empty:
        raise ValueError(f"No scores for the selected metric: {metric_name}")

    bar_chart = bin_bar_chart(
        binned_df, metric_name, scope, show_decomposition=show_decomposition, color_params=color_params
    )
    line_chart = performance_rate_line_chart(
        bar_chart, metric_name, scope, show_decomposition=show_decomposition, color_params=color_params
    )
    mean_rule = performance_average_rule(binned_df[BinSchema.indicator].mean(), scope, color_params=color_params)

    chart_composition: alt_api.LayerChart = bar_chart + line_chart + mean_rule

    title = CHART_TITLES[scope]
    chart_composition = chart_composition.encode(alt.X(title=metric_name.title()), alt.Y(title=title)).properties(
        title=title
    )
    return chart_composition
