from typing import List, Literal, TypedDict

from encord_active_components.renderer import Components, render


class GroupedTags(TypedDict):
    data: List[str]
    label: List[str]


Scope = Literal["data_quality", "label_quality", "model_quality"]


def explorer(project_name: str, items: List[str], scope: Scope):
    return render(
        component=Components.EXPLORER,
        props={"projectName": project_name, "items": items, "scope": scope},
    )


if __name__ == "__main__":
    pass
