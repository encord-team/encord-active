from dataclasses import dataclass
from typing import Optional

import pandas as pd


def filter_tags(df, tag_filters):
    if not tag_filters:
        return df
    filtered_rows = [True if set(tag_filters) <= set(x) else False for x in df["tags"]]
    return df.loc[filtered_rows]


@dataclass
class FilterConfig:
    data_tags: Optional[list[str]] = None
    label_tags: Optional[list[str]] = None

    def filter_df(self, df: pd.DataFrame):
        filtered_df = df.copy()
        filtered_df = filter_tags(filtered_df, self.data_tags)
        filtered_df = filter_tags(filtered_df, self.label_tags)
        return filtered_df

