from typing import Dict, List, Set, TypedDict

import pandas as pd

from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, TagScope
from encord_active.lib.project.project_file_structure import ProjectFileStructure


class GroupedTags(TypedDict):
    data: List[str]
    label: List[str]


def scoped_tags(tags: List[Tag], scopes: Set[TagScope]) -> List[Tag]:
    return [tag for tag in tags if tag.scope in scopes]


def all_tags(project_file_structure: ProjectFileStructure) -> List[Tag]:
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


def populate_tags_with_nested_classifications(pfs: ProjectFileStructure, option_answer_hashes: set[str]):
    with DBConnection(pfs) as conn:
        mm = MergedMetrics(conn).all()
    no_label_tag = Tag("no classification", TagScope.LABEL)

    def update_label_tags(label: dict, base_key: str, identifier_key: str, answer_key: str, answer_dict: dict):
        label_identifier = label[identifier_key]
        label_hash = label[answer_key]
        key = f"{base_key}_{label_identifier}"

        if key not in mm.index:
            return

        tags = mm.loc[key].tags
        for question in answer_dict[label_hash].get("classifications", [{}]):
            if question["featureHash"] not in option_answer_hashes:
                continue

            name = question["name"]
            for answer in question["answers"]:
                value = answer["name"]
                tag = Tag(f"{name}: {value}", TagScope.LABEL)
                if tag not in tags:
                    tags.append(tag)

        if len(tags) == 0:
            tags.append(no_label_tag)

        with DBConnection(pfs) as conn:
            MergedMetrics(conn).update_tags(key, tags)

    iterator = DatasetIterator(pfs.project_dir)
    for du, _ in iterator.iterate(desc="Tagging Objects with nested classifications"):
        base_key = f"{iterator.label_hash}_{iterator.du_hash}_{iterator.frame:05d}"
        object_answers = iterator.label_rows[iterator.label_hash]["object_answers"]
        classification_answers = iterator.label_rows[iterator.label_hash]["classification_answers"]

        for clf in du.get("labels", {}).get("classifications", []):
            update_label_tags(clf, base_key, "featureHash", "classificationHash", classification_answers)

        for obj in du.get("labels", {}).get("objects", []):
            update_label_tags(obj, base_key, "objectHash", "objectHash", object_answers)
