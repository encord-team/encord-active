from sqlite3 import OperationalError
from typing import Callable, List

from encord_active.lib.db.connection import DBConnection

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
        stripped = tag.strip()
        if not stripped:
            raise ValueError("Empty tags are not allowed")

        if tag in self.all():
            raise ValueError("Tag already exists")

        with DBConnection() as conn:
            return conn.execute(f"INSERT INTO {TABLE_NAME} (label) VALUES('{tag}')")
