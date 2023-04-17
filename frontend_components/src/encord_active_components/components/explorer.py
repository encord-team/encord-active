from enum import Enum
from typing import List, Literal, Optional, Tuple, TypedDict

from encord_active_components.renderer import Components, render


class GroupedTags(TypedDict):
    data: List[str]
    label: List[str]


class GalleryItem(TypedDict):
    id: str


EmbeddingType = Literal["classification", "object", "image"]
Scope = Literal["data_quality", "label_quality", "model_quality"]


class OutputAction(str, Enum):
    CHANGE_PAGE = "CHANGE_PAGE"


Output = Tuple[OutputAction, Optional[int]]


def explorer(project_name: str, items: List[str], scope: Scope, embeddings_type: EmbeddingType) -> Output:
    return render(
        component=Components.EXPLORER,
        props={
            "projectName": project_name,
            "items": items,
            "scope": scope,
            "embeddingsType": embeddings_type,
        },
    )


if __name__ == "__main__":
    pass
