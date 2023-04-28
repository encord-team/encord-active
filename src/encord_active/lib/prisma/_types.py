from typing import Any, Callable, Coroutine, Tuple, Type, TypeVar

from pydantic import BaseModel
from typing_extensions import Literal as Literal
from typing_extensions import Protocol as Protocol
from typing_extensions import TypedDict as TypedDict
from typing_extensions import TypeGuard as TypeGuard
from typing_extensions import get_args as get_args
from typing_extensions import runtime_checkable as runtime_checkable

Method = Literal["GET", "POST"]

CallableT = TypeVar("CallableT", bound="FuncType")
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)

# TODO: use a TypeVar everywhere
FuncType = Callable[..., object]
CoroType = Callable[..., Coroutine[Any, Any, object]]


@runtime_checkable
class InheritsGeneric(Protocol):
    __orig_bases__: Tuple["_GenericAlias"]


class _GenericAlias(Protocol):
    __origin__: Type[object]
