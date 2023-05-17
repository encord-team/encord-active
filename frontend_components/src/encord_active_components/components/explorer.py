from typing import List, Literal, Optional

from encord_active_components.renderer import Components, render

Scope = Literal["data_quality", "label_quality", "model_quality"]


def explorer(auth_token: Optional[str], project_name: str, items: List[str], scope: Scope, api_url: str):
    return render(
        component=Components.EXPLORER,
        props={
            "authToken": auth_token,
            "projectName": project_name,
            "items": items,
            "scope": scope,
            "baseUrl": api_url,
        },
    )
