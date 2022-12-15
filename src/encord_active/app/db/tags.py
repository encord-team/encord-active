from enum import Enum
from sqlite3 import OperationalError
from typing import Callable, List, NamedTuple

import streamlit as st

from encord_active.app.common.state import ALL_TAGS
from encord_active.app.db.connection import DBConnection

TABLE_NAME = "tags"


class TagScope(str, Enum):
    DATA = "Data"
    LABEL = "Label"
    PREDICTION = "Prediction"


class Tag(NamedTuple):
    name: str
    scope: str


def ensure_existence(fn: Callable):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except OperationalError:
            with DBConnection() as conn:
                conn.execute(
                    f"""
                     CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        scope TEXT NOT NULL
                     )
                     """
                )
                all_tags = st.session_state.get(ALL_TAGS)
                if all_tags:
                    conn.executemany(f"INSERT INTO {TABLE_NAME} (name, scope) VALUES(?, ?) ", all_tags)

            return fn(*args, **kwargs)

    return wrapper


class Tags(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Tags, cls).__new__(cls)

        return cls.instance

    @ensure_existence
    def all(self) -> List[Tag]:
        with DBConnection() as conn:
            return [
                Tag(name, scope) for name, scope, in conn.execute(f"SELECT name, scope FROM {TABLE_NAME}").fetchall()
            ]

    @ensure_existence
    def create_tag(self, tag: Tag):
        with DBConnection() as conn:
            return conn.execute(f"INSERT INTO {TABLE_NAME} (name, scope) VALUES(?, ?) ", tag)
