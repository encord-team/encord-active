from enum import Enum
from os import environ
from pathlib import Path

from streamlit.components.v1.components import declare_component

_RELEASE = environ.get("RELEASE", "TRUE")

if _RELEASE == "TRUE":
    build_dir = (Path(__file__).parent / "frontend/dist").expanduser().resolve()
    render = declare_component("frontend_components", path=build_dir.as_posix())
else:
    render = declare_component("frontend_components", url="http://localhost:5173")


class Components(str, Enum):
    PAGES_MENU = "PagesMenu"
    PROJECT_SELECTION = "ProjectsPage"
    EXPLORER = "Explorer"
