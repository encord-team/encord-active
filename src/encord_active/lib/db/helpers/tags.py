from typing import Dict, List, Set, TypedDict

import pandas as pd

from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, TagScope
from encord_active.lib.project.project_file_structure import ProjectFileStructure


class GroupedTags(TypedDict):
    data: List[str]
    label: List[str]


def scoped_tags(tags: List[Tag], scopes: Set[TagScope]) -> List[Tag]:
    return [tag for tag in tags if tag.scope in scopes]


def all_tags(project_file_structure: ProjectFileStructure):
    with DBConnection(project_file_structure) as conn:
        tag_list = MergedMetrics(conn).all(columns=["tags"])["tags"].values
    return list({tag for tag_list in tag_list for tag in tag_list})


def count_of_tags(tag_list: List[Tag], df: pd.DataFrame) -> Dict[str, int]:
    if not tag_list:
        return {}

    tag_counts = df["tags"].value_counts()

    total_tags_count: Dict[str, int] = {tag.name: 0 for tag in tag_list}
    for unique_list, count in tag_counts.items():
        for tag in unique_list:
            total_tags_count[tag.name] += count

    return total_tags_count


def to_grouped_tags(tags: List[Tag]) -> GroupedTags:
    grouped_tags = GroupedTags(data=[], label=[])

    for name, scope in tags:
        if scope == TagScope.DATA:
            grouped_tags["data"].append(name)
        elif scope == TagScope.LABEL:
            grouped_tags["label"].append(name)

    return grouped_tags


def from_grouped_tags(tags: GroupedTags) -> tuple[List[Tag], List[Tag]]:
    data_tags = [Tag(tag, TagScope.DATA) for tag in tags["data"]]
    label_tags = [Tag(tag, TagScope.LABEL) for tag in tags["label"]]
    return data_tags, label_tags
