from enum import Enum
from typing import List, Optional, Tuple, TypedDict, Union

from encord_active_components.components.projects_page import Project, ProjectStats
from encord_active_components.renderer import Components, render


class MenuItem(TypedDict):
    key: str
    label: str
    children: Optional[List["MenuItem"]]


class OutputAction(str, Enum):
    VIEW_ALL_PROJECTS = "VIEW_ALL_PROJECTS"
    SELECT_PROJECT = "SELECT_PROJECT"
    SELECT_PAGE = "SELECT_PAGE"


def pages_menu(
    items: List[MenuItem], projects: List[Project] = [], selected_project_hash: Optional[str] = None
) -> Tuple[OutputAction, Union[str, None]]:
    return render(
        component=Components.PAGES_MENU,
        props={"items": items, "projects": projects, "selectedProjectHash": selected_project_hash},
    )


if __name__ == "__main__":
    ITEMS: List[MenuItem] = [
        {
            "key": "Example 1",
            "label": "Example 1",
            "children": [
                {"key": "Example 1#Summary", "label": "Summary", "children": None},
                {"key": "Example 1#Explorer", "label": "Explorer", "children": None},
            ],
        },
        {
            "key": "Example 2 ",
            "label": "Example 2 ",
            "children": [
                {"key": "Example 2 #Summary", "label": "Summary", "children": None},
                {"key": "Example 2 #Explorer", "label": "Explorer", "children": None},
            ],
        },
    ]
    PROJECTS = [
        Project(
            name="Foo",
            hash="d3d81fb8-634c-4909-be57-49f94adc93dd",
            downloaded=True,
            stats=ProjectStats(dataUnits=1000, labels=14566, classes=8),
            imageUrl="",
        ),
        Project(
            name="Bar",
            hash="603336c6-c5c4-4ae9-87a7-216e5201ede5",
            downloaded=True,
            stats=ProjectStats(dataUnits=100, labels=166, classes=2),
            imageUrl="",
        ),
    ]
    key, action = pages_menu(ITEMS, PROJECTS)
