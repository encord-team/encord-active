from enum import Enum
from sqlite3 import OperationalError
from typing import Callable, List, NamedTuple

from encord_active.lib.db.connection import DBConnection
from encord_active.lib.metrics.utils import MetricScope

TABLE_NAME = "tags"


class TagScope(str, Enum):
    DATA = "Data"
    LABEL = "Label"


class Tag(NamedTuple):
    name: str
    scope: TagScope


SCOPE_EMOJI = {
    TagScope.DATA.value: "🖼️",
    TagScope.LABEL.value: "✏️",
}

METRIC_SCOPE_TAG_SCOPES = {
    MetricScope.DATA_QUALITY: {TagScope.DATA},
    MetricScope.LABEL_QUALITY: {TagScope.DATA, TagScope.LABEL},
    MetricScope.MODEL_QUALITY: {TagScope.DATA},
}


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
        stripped = tag.name.strip()
        if not stripped:
            raise ValueError("Empty tags are not allowed")

        if tag in self.all():
            raise ValueError("Tag already exists")

        with DBConnection() as conn:
            return conn.execute(f"INSERT INTO {TABLE_NAME} (name, scope) VALUES(?, ?) ", tag)
