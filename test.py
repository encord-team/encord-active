from pathlib import Path

import streamlit as st

from encord_active.app.common.state_new import use_state


def foo():
    value, set_value = use_state(0, key="foo")

    st.experimental_show(value())

    set_value(8)
    st.experimental_show(value())


foo()
