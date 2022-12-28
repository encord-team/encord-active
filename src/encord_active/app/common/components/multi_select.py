import logging
from typing import List, Optional, cast

import streamlit as st
from natsort import natsorted

from encord_active.app.common.state_new import create_key

logger = logging.getLogger(__name__)


def multiselect_with_all_option(
    label: str,
    options: List[str],
    custom_all_name: str = "All",
    key: Optional[str] = None,
    default: Optional[List[str]] = None,
    **kwargs,
):
    if "on_change" in kwargs:
        logger.warning("`st.multiselect.on_change` is being overwritten by custom multiselect with All option")

    key = key or create_key()

    default_list = [custom_all_name] if default is None else default

    # Filter classes
    def on_change(key: str, prev: List[str] = default_list):
        next: List[str] = (st.session_state.get(key) or default_list).copy()

        if custom_all_name not in prev and custom_all_name in next:
            next = [custom_all_name]
        if len(next) > 1 and custom_all_name in next:
            next.remove(custom_all_name)

        st.session_state[key] = next

    sorted_options = cast(List[str], natsorted(list(options)))
    options = [custom_all_name] + sorted_options

    current: List[str] = st.session_state.get(key) or default_list

    return st.multiselect(
        label, options, key=key, on_change=on_change, args=(current, key), default=default_list, **kwargs
    )
