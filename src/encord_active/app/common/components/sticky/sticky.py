from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html


def sticky_header():
    container = st.container()
    container.markdown('<div id="sticky"/>', unsafe_allow_html=True)

    with container:
        with open(Path(__file__).parent / "sticky.html", encoding="utf-8") as f:
            html(f.read(), height=0, width=0)

    return container
