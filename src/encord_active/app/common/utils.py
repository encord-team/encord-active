from pathlib import Path

import streamlit as st

from encord_active.app.common.css import write_page_css


def set_page_config():
    favicon_pth = Path(__file__).parents[1] / "assets" / "favicon-32x32.png"
    st.set_page_config(
        page_title="Encord Active",
        layout="wide",
        page_icon=favicon_pth.as_posix(),
    )


def setup_page():
    write_page_css()
