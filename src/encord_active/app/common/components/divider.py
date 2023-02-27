import streamlit as st


def divider(width=1, color="#D6D6D8"):
    st.markdown(
        f"<hr style='height:{width}px;border:none;color:{color};background-color:{color};' /> ",
        unsafe_allow_html=True,
    )
