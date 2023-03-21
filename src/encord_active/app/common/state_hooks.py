import inspect
from typing import Callable, Generic, Optional, TypeVar, Union, overload

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

    return UseState(value, key)


class UseState(Generic[T]):
    def __init__(self, initial: Optional[T] = None, key: Optional[str] = None) -> None:
        self.key = key or create_key()
        st.session_state.setdefault(SCOPED_STATES, {}).setdefault(self.key, initial)

    @overload
    def set(self, arg: T):
        ...

    @overload
    def set(self, arg: Reducer[T]):
        ...

    def set(self, arg: Union[T, Reducer[T]]):
        if callable(arg):
            st.session_state[SCOPED_STATES][self.key] = arg(st.session_state[SCOPED_STATES][self.key])
        else:
            st.session_state[SCOPED_STATES][self.key] = arg

    @property
    def value(self) -> T:
        return st.session_state[SCOPED_STATES][self.key]
