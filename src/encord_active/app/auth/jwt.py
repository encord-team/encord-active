from typing import Optional

import streamlit as st
from jwt import decode  # type: ignore

from encord_active.app.common.state_hooks import UseState
from encord_active.server.settings import settings


def auth():
    if not settings.JWT_SECRET:
        return

    token_state = UseState[Optional[str]](None, clearable=False)

    try:
        if token_state.value is None:
            query_params = st.experimental_get_query_params()
            token, *_ = query_params["token"]
            token_state.set(token)
        else:
            token = token_state.value

        decoded = decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        assert decoded["deployment_name"] == settings.DEPLOYMENT_NAME
        st.experimental_set_query_params()
    except:
        st.error("Unauthorized")
        st.stop()
