import inspect
from typing import Callable, Optional, TypeVar, Union, overload

import streamlit as st

T = TypeVar("T")
Reducer = Callable[[T], T]

SCOPED_STATES = "scoped_states"


def create_key():
    stk = inspect.stack()
    frame = stk[2]
    return f"{frame.filename}:{frame.function}:{frame.lineno}"


def use_memo(initial: Callable[[], T], key: Optional[str] = None):
    key = key or create_key()
    st.session_state.setdefault(SCOPED_STATES, {})

    if key not in st.session_state[SCOPED_STATES]:
        st.session_state[SCOPED_STATES][key] = initial()

    value: T = st.session_state[SCOPED_STATES][key]
    return value


def use_lazy_state(initial: Callable[[], T], key: Optional[str] = None):
    key = key or create_key()
    st.session_state.setdefault(SCOPED_STATES, {})

    if key not in st.session_state[SCOPED_STATES]:
        st.session_state[SCOPED_STATES][key] = initial()
    value: T = st.session_state[SCOPED_STATES][key]

    return use_state(value, key)


# TODO: is there a way to make this work and have proper types?
# def use_state(initial: Union[T, Callable[[], T]], key: Optional[str] = create_key()):
#     st.session_state.setdefault(SCOPED_STATES, {})
#     st.session_state[SCOPED_STATES].setdefault(key, initial() if callable(initial) else initial)
def use_state(initial: T, key: Optional[str] = None):
    key = key or create_key()
    st.session_state.setdefault(SCOPED_STATES, {}).setdefault(key, initial)

    @overload
    def set_state(arg: T):
        ...

    @overload
    def set_state(arg: Reducer[T]):
        ...

    def set_state(arg: Union[T, Reducer[T]]):
        if callable(arg):
            st.session_state[SCOPED_STATES][key] = arg(st.session_state[SCOPED_STATES][key])
        else:
            st.session_state[SCOPED_STATES][key] = arg

    def get_state() -> T:
        return st.session_state[SCOPED_STATES][key]

    return get_state, set_state
