from typing import Optional

from encord_active_components.renderer import Components, render


def active(project_hash: str, api_url: str, auth_token: Optional[str] = None):
    return render(
        component=Components.ACTIVE,
        props={
            # "authToken": auth_token,
            "projectHash": project_hash,
            "baseUrl": api_url,
        },
    )
