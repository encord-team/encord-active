from pathlib import Path

import streamlit as st

from encord_active.app.common.state_new import use_state


def foo():
    value, set_value = use_state(0)
    set_value(lambda prev: prev + 1)

    st.write(value())

    set_value(lambda prev: prev + 5)

    st.write(value())


foo()
