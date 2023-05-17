from typing import Optional

import streamlit as st
from jwt import decode  # type: ignore

from encord_active.app.common.state_hooks import UseState
from encord_active.server.settings import get_settings

AUTH_TOKEN_STATE_KEY = "auth_token"


def get_auth_token():
    token_state = UseState[Optional[str]](None, clearable=False, key=AUTH_TOKEN_STATE_KEY)
    return token_state.value


def auth():
    settings = get_settings()
    if not settings.JWT_SECRET:
        return

    token_state = UseState[Optional[str]](None, clearable=False, key=AUTH_TOKEN_STATE_KEY)

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
