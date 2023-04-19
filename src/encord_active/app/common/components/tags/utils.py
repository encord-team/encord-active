from typing import List, Set

from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, TagScope


def scoped_tags(scopes: Set[TagScope]) -> List[Tag]:
    return [tag for tag in all_tags() if tag.scope in scopes]


def all_tags():
    tag_list = MergedMetrics().all(columns=["tags"])["tags"].values
    return list({tag for tag_list in tag_list for tag in tag_list})
