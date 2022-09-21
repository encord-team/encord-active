import streamlit.components.v1 as components

from encord_active.app.conf import FRONTEND

_pages_menu = components.declare_component(
    "pages-menu",
    url=f"{FRONTEND}/pages-menu",
)


def pages_menu(items) -> str:
    return _pages_menu(items=items)
