from typing import Optional

import altair as alt
import pandas as pd


def get_histogram(df: pd.DataFrame, column_name: str, metric_name: Optional[str] = None):
    chart_title = "Data distribution"

    if metric_name:
        chart_title += f" - {metric_name}"
    else:
        metric_name = column_name

    bar_chart = (
        alt.Chart(df, title=chart_title)
        .mark_bar()
        .encode(
            alt.X(f"{column_name}:Q", bin=alt.Bin(maxbins=100), title=metric_name),
            alt.Y("count()", title="Num. samples"),
            tooltip=[
                alt.Tooltip(f"{column_name}:Q", title=metric_name, format=",.3f", bin=True),
                alt.Tooltip("count():Q", title="Num. samples", format="d"),
            ],
        )
        .properties(height=200)
    )
    return bar_chart
