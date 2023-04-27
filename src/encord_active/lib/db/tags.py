from enum import Enum
from typing import NamedTuple

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
