import mimetypes
from math import floor, log
from pathlib import Path

import streamlit as st

from encord_active.app.common.css import write_page_css


# NOTE: this ensures mimetypes for custom components are always loaded
def add_mimetypes():
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")


def set_page_config():
    add_mimetypes()
    favicon_pth = Path(__file__).parents[1] / "assets" / "favicon-32x32.png"
    st.set_page_config(
        page_title="Encord Active",
        layout="wide",
        page_icon=favicon_pth.as_posix(),
    )


def setup_page():
    write_page_css()


def human_format(number: int) -> str:
    units = ["", "K", "M", "G", "T", "P"]
    k = 1000.0

    if number == 0:
        return "0"

    magnitude = int(floor(log(abs(number), k)))

    unit = units[magnitude]
    fmt = "{:.2f}{}" if unit else "{:.0f}{}"
    return fmt.format(number / k**magnitude, units[magnitude])
