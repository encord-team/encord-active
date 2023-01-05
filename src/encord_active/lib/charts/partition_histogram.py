import altair as alt
import pandas as pd


def get_partition_histogram(balanced_df: pd.DataFrame, metric: str):
    return (
        alt.Chart(balanced_df)
        .mark_bar(
            binSpacing=0,
        )
        .encode(
            x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=50)),
            y="count()",
            color="partition:N",
            tooltip=["partition", "count()"],
        )
    )
