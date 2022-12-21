from typing import Dict

import pandas as pd

from encord_active.lib.db.tags import Tags


def count_of_tags(df: pd.DataFrame) -> Dict[str, int]:
    tag_list = Tags().all()
    if not tag_list:
        return {}

    tag_counts = df["tags"].value_counts()

    total_tags_count: Dict[str, int] = {tag.name: 0 for tag in tag_list}
    for unique_list, count in tag_counts.items():
        for tag in unique_list:
            total_tags_count[tag.name] += count

    return total_tags_count
