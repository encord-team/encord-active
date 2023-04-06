import altair as alt
import pandas as pd


def get_histogram(df: pd.DataFrame, metric_name: str):
    return (
        alt.Chart(df, title=f"Data distribution - {metric_name}")
        .mark_bar()
        .encode(
            alt.X(f"{metric_name}:Q", bin=alt.Bin(maxbins=100), title=metric_name),
            alt.Y("count()", title="Num. samples"),
            tooltip=[
                alt.Tooltip(f"{metric_name}:Q", title=metric_name, format=",.3f", bin=True),
                alt.Tooltip("count():Q", title="Num. samples", format="d"),
            ],
        )
        .properties(height=200)
    )
