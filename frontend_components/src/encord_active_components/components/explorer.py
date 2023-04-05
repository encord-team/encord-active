from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict

from encord_active_components.renderer import Components, render

# class GalleryItemType(str, Enum):
#     DATA:


class Metadata(TypedDict):
    annotator: Optional[str]
    labelClass: Optional[str]
    metrics: Dict[str, str]


class GroupedTags(TypedDict):
    data: List[str]
    label: List[str]


class GalleryItem(TypedDict):
    id: str
    url: str
    editUrl: str
    tags: GroupedTags
    metadata: Metadata


class PaginationInfo(TypedDict):
    current: int
    total: int


class OutputAction(str, Enum):
    CHANGE_PAGE = "CHANGE_PAGE"


Output = Tuple[OutputAction, Optional[int]]


def explorer(items: List[GalleryItem], all_tags: GroupedTags) -> Output:
    return render(component=Components.EXPLORER, props={"items": items, "tags": all_tags})


if __name__ == "__main__":
    pass
    # key = pages_menu(ITEMS)
