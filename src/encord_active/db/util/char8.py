from sqlalchemy import Column, TypeDecorator
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.type_api import TypeEngine
from sqlalchemy.types import BigInteger

__all__ = ['Char8']


class Char8(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.

    """

    impl = BigInteger
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        return dialect.type_descriptor(BigInteger())

    def bind_processor(self, dialect: Dialect):
        def pack_to_int(value: str) -> int:
            if len(value) != 8:
                raise ValueError(f"Char8 has incorrect length: {value}")
            value_bytes = value.encode("ascii")
            return int.from_bytes(value_bytes, byteorder="little", signed=True)
        return pack_to_int

    def result_processor(self, dialect: Dialect, coltype):
        def unpack_from_int(value: int) -> str:
            value_bytes = value.to_bytes(length=8, byteorder="little", signed=True)
            return value_bytes.decode("ascii")
        return unpack_from_int
