from typing import List, Optional, TypedDict

from encord_active_components.renderer import Components, render


class MenuItem(TypedDict):
    key: str
    label: str
    children: Optional[List["MenuItem"]]


def pages_menu(items: List[MenuItem], selected: Optional[str] = None) -> str:
    return render(component=Components.PAGES_MENU, props={"items": items, "selected": selected})


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
    key = pages_menu(ITEMS)
