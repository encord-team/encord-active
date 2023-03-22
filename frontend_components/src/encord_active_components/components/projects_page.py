from enum import Enum
from typing import List, Optional, Tuple, TypedDict

from encord_active_components.renderer import Components, render


class ProjectStats(TypedDict):
    dataUnits: int
    labels: int
    classes: int


class Project(TypedDict):
    name: str
    hash: str
    imageUrl: str
    downloaded: bool
    stats: Optional[ProjectStats]


class OutputAction(str, Enum):
    SELECT_SANDBOX_PROJECT = "SELECT_SANDBOX_PROJECT"
    SELECT_USER_PROJECT = "SELECT_USER_PROJECT"
    IMPORT_ENCORD = "IMPORT_ENCORD"
    IMPORT_COCO = "IMPORT_COCO"
    INIT = "INIT"


def projects_page(
    user_projects: List[Project] = [], sandbox_projects: List[Project] = []
) -> Optional[Tuple[OutputAction, str]]:
    return render(
        component=Components.PROJECT_SELECTION,
        props={"userProjects": user_projects, "sandboxProjects": sandbox_projects},
    )


if __name__ == "__main__":
    pass
    # key = pages_menu(ITEMS)
