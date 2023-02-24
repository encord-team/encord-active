from os import environ
from pathlib import Path
from typing import List, Optional, TypedDict

import streamlit.components.v1 as components

_RELEASE = environ.get("RELEASE", "TRUE")


class MenuItem(TypedDict):
    key: str
    label: str
    children: Optional[List["MenuItem"]]


if _RELEASE == "TRUE":
    build_dir = (Path(__file__).parent / "frontend/dist").expanduser().resolve()
    _pages_menu = components.declare_component("pages_menu", path=build_dir.as_posix())
else:
    _pages_menu = components.declare_component("pages-menu", url="http://localhost:5173")


def pages_menu(items: List[MenuItem]) -> str:
    return _pages_menu(items=items)


if __name__ == "__main__":
    items: List[MenuItem] = [
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
    key = pages_menu(items)
