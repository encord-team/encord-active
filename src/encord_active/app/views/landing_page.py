from pathlib import Path

import streamlit as st

from encord_active.app.common.utils import set_page_config, setup_page


def landing_page():
    setup_page()

    logo_pth = Path(__file__).parents[1] / "assets" / "encord_2_02.png"
    _, col, _ = st.columns([4, 2, 4])  # Hack to center and scale content
    with col:
        st.image(logo_pth.as_posix())

    _, col, _ = st.columns([2, 6, 2])

    with col:
        st.video("https://youtu.be/C0h_HqX1WxU")

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown((Path(__file__).parent.parent / "static" / "landing-page-intro.md").read_text())
    with right_col:
        st.markdown((Path(__file__).parent.parent / "static" / "landing-page-documentation.md").read_text())


if __name__ == "__main__":
    set_page_config()
    landing_page()
