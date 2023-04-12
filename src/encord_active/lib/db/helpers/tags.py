from typing import Dict, List

import pandas as pd
from encord_active_components.components.explorer import GroupedTags

from encord_active.lib.db.tags import Tag, Tags, TagScope


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


def to_grouped_tags(tags: List[Tag]):
    grouped_tags = GroupedTags(data=[], label=[])

    for name, scope in tags:
        if scope == TagScope.DATA:
            grouped_tags[scope.lower()].append(name)
        elif scope == TagScope.LABEL:
            grouped_tags["label"].append(name)

    return grouped_tags
