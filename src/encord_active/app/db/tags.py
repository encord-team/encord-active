from sqlite3 import OperationalError
from typing import Callable, List

import streamlit as st

from encord_active.app.common.state import ALL_TAGS
from encord_active.app.db.connection import DBConnection

TABLE_NAME = "tags"


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
                        label TEXT NOT NULL
                     )
                     """
                )
                all_tags = st.session_state.get(ALL_TAGS)
                if all_tags:
                    conn.executemany(f" INSERT INTO {TABLE_NAME} (label) VALUES(?) ", [all_tags])

            return fn(*args, **kwargs)

    return wrapper


class Tags(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Tags, cls).__new__(cls)

        return cls.instance

    @ensure_existence
    def all(self) -> List[str]:
        with DBConnection() as conn:
            return [tag for tag, in conn.execute(f"SELECT label FROM {TABLE_NAME}").fetchall()]

    @ensure_existence
    def create_tag(self, tag: str):
        with DBConnection() as conn:
            return conn.execute(f"INSERT INTO {TABLE_NAME} (label) VALUES('{tag}')")
