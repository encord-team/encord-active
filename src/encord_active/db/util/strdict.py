from sqlalchemy import JSON, TypeDecorator
from sqlalchemy.dialects.postgresql import HSTORE
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.type_api import TypeEngine

__all__ = ["StrDict"]


class StrDict(TypeDecorator):
    """Platform-independent HSTORE type.

    Uses PostgreSQL's HSTORE type, otherwise uses
    JSONB, storing as a dictionary of string to string.
    """

    impl = HSTORE
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(HSTORE())  # type: ignore
        else:
            return dialect.type_descriptor(JSON())  # type: ignore
