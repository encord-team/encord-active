from typing import Any, Literal, Optional

from encord_active_components.renderer import Components, render

Scope = Literal["data_quality", "label_quality", "model_quality"]


# NOTE: typing filters as any for now to not spend time on setting up a way to
# share the definition of filters. this is temp until we kill streamlit.
def explorer(auth_token: Optional[str], project_name: str, scope: Scope, api_url: str, filters: Optional[Any] = None):
    return render(
        component=Components.EXPLORER,
        props={
            "authToken": auth_token,
            "projectName": project_name,
            "scope": scope,
            "baseUrl": api_url,
            "filters": filters,
        },
    )
