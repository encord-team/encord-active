from typing import List, Literal

from encord_active_components.renderer import Components, render

Scope = Literal["data_quality", "label_quality", "model_quality"]


def explorer(project_name: str, items: List[str], scope: Scope):
    return render(
        component=Components.EXPLORER,
        props={"projectName": project_name, "items": items, "scope": scope},
    )
