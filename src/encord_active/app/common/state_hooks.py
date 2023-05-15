import inspect
from typing import Callable, Generic, Optional, Tuple, TypeVar, Union, overload

import streamlit as st

from encord_active.app.common.state import StateKey

T = TypeVar("T")
Reducer = Callable[[T], T]


def create_key():
    frames = inspect.stack()
    frame_keys = [
        f"{frame.filename}:{frame.function}:{frame.lineno}" for frame in frames if "encord_active" in frame.filename
    ]
    return str(hash("&".join(frame_keys)))


def use_memo(initial: Callable[[], T], key: Optional[str] = None, clearable: bool = True) -> Tuple[T, Callable[[], T]]:
    key = key or create_key()
    scope = st.session_state.setdefault(StateKey.MEMO, {}) if clearable else st.session_state

    if key not in scope:
        scope[key] = initial()

    def refresh() -> T:
        scope[key] = initial()
        return scope[key]

    return scope[key], refresh


def use_lazy_state(initial: Callable[[], T], key: Optional[str] = None):
    key = key or create_key()
    scope = st.session_state.setdefault(StateKey.SCOPED, {})

    if key not in scope:
        scope[key] = initial()
    value: T = scope[key]

    return UseState(value, key)


class UseState(Generic[T]):
    def __init__(self, initial: T, key: Optional[str] = None, clearable=True) -> None:
        self._initial = initial
        self._key = key or create_key()
        self._scope = StateKey.SCOPED_AND_PERSISTED if clearable else StateKey.SCOPED
        st.session_state.setdefault(self._scope, {}).setdefault(self._key, initial)

    @overload
    def set(self, arg: T):
        ...

    @overload
    def set(self, arg: Reducer[T]):
        ...

    def set(self, arg: Union[T, Reducer[T]]):
        if callable(arg):
            new_value = arg(st.session_state[self._scope][self._key])
        else:
            new_value = arg

        if new_value == self.value:
            return

        st.session_state.setdefault(self._scope, {})[self._key] = new_value

    @property
    def value(self) -> T:
        return st.session_state.get(self._scope, {}).get(self._key, self._initial)
