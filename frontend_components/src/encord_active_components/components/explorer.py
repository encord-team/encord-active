from enum import Enum
from typing import List, Optional, Tuple, TypedDict

from encord_active_components.renderer import Components, render


class GroupedTags(TypedDict):
    data: List[str]
    label: List[str]


class GalleryItem(TypedDict):
    id: str


class EmbeddingType(str, Enum):
    CLASSIFICATION = "classification"
    OBJECT = "object"
    IMAGE = "image"


class OutputAction(str, Enum):
    CHANGE_PAGE = "CHANGE_PAGE"


Output = Tuple[OutputAction, Optional[int]]


def explorer(project_name: str, items: List[str], all_tags: GroupedTags, embeddings_type: EmbeddingType) -> Output:
    return render(
        component=Components.EXPLORER,
        props={"projectName": project_name, "items": items, "tags": all_tags, "embeddingsType": embeddings_type},
    )


if __name__ == "__main__":
    pass
