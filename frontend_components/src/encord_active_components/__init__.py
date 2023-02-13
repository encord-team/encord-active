from os import environ
from pathlib import Path
from typing import List, Optional, TypedDict

import streamlit.components.v1 as components

_RELEASE = environ.get("RELEASE", "FALSE")


class MenuItem(TypedDict):
    key: str
    label: str
    children: Optional[List["MenuItem"]]


if _RELEASE.lower() == "true":
    _pages_menu = components.declare_component("pages-menu", url="http://localhost:5173")
else:
    build_dir = (Path(__file__).parent / "frontend/dist").expanduser().resolve()
    _pages_menu = components.declare_component("pages_menu", path=build_dir.as_posix())


def pages_menu(items: List[MenuItem]) -> str:
    return _pages_menu(items=items)
