from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict

from encord_active_components.renderer import Components, render


class Metadata(TypedDict):
    annotator: Optional[str]
    labelClass: Optional[str]
    metrics: Dict[str, str]


class GroupedTags(TypedDict):
    data: List[str]
    label: List[str]


class GalleryItem(TypedDict):
    id: str
    editUrl: str
    tags: GroupedTags
    metadata: Metadata


class OutputAction(str, Enum):
    CHANGE_PAGE = "CHANGE_PAGE"


Output = Tuple[OutputAction, Optional[int]]


def explorer(project_name: str, items: List[GalleryItem], all_tags: GroupedTags) -> Output:
    return render(component=Components.EXPLORER, props={"projectName": project_name, "items": items, "tags": all_tags})


if __name__ == "__main__":
    pass
