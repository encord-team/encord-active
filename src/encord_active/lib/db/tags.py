from enum import Enum
from sqlite3 import OperationalError
from typing import Callable, List, NamedTuple

from encord_active.lib.db.connection import DBConnection

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
