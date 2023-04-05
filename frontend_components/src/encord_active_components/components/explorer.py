from enum import Enum
from typing import Dict, List, Optional, TypedDict

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


# class OutputAction(str, Enum):
#     INIT = "INIT"


def explorer(items: List[GalleryItem], all_tags: GroupedTags):
    return render(component=Components.EXPLORER, props={"items": items, "tags": all_tags})


if __name__ == "__main__":
    pass
    # key = pages_menu(ITEMS)
