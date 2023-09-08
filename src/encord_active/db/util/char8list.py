from typing import Union

import numpy as np
from psycopg2.extensions import Binary
from sqlalchemy import BigInteger, TypeDecorator
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.type_api import TypeEngine
from sqlalchemy.types import BINARY
from sqlmodel import AutoString

__all__ = ["Char8List"]


class Char8List(TypeDecorator):
    """Platform-independent Char8 list type.

    Uses PostgreSQL's bigint[] type, otherwise
    uses a comma separated string of raw values.
    """

    impl = ARRAY(BigInteger)
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        self.dim = dim

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(ARRAY(BigInteger))  # type: ignore
        else:
            return dialect.type_descriptor(AutoString())  # type: ignore

    def bind_processor(self, dialect: Dialect):
        def as_validated_ndarray(value: Union[np.ndarray, bytes]) -> np.ndarray:
            np_value = np.frombuffer(value, dtype=np.float64) if isinstance(value, bytes) else value
            if np_value.dtype != np.float64:
                raise ValueError(f"pgvector with wrong type: {np_value.dtype}")
            if np_value.ndim != 1 or np_value.shape != (self.dim,):
                raise ValueError(f"pgvector with wrong dimension: {np_value.shape}")
            return np_value

        if dialect.name == "postgresql":

            def process_pg(value: Union[np.ndarray, bytes]) -> str:
                np_value = as_validated_ndarray(value)
                return "[" + ",".join([str(float(v)) for v in np_value]) + "]"

            return process_pg
        else:
            bind_other = BINARY().bind_processor(dialect)

            def process_fallback(value: Union[np.ndarray, bytes]) -> bytes:
                np_value = as_validated_ndarray(value)
                return bind_other(np_value.tobytes(order="C"))  # type: ignore

            return process_fallback

    def result_processor(self, dialect: Dialect, coltype):
        result_processor = BINARY().result_processor(dialect, coltype)

        def process(value: Union[np.ndarray, str, bytes, Binary]) -> bytes:
            if isinstance(value, str):
                np_value = np.array(value[1:-1].split(","), dtype=np.float64)
            elif isinstance(value, np.ndarray):
                np_value = value
            elif isinstance(value, bytes):
                np_value = np.frombuffer(value, dtype=np.float64)
            else:
                result_value = result_processor(value)
                np_value = np.frombuffer(result_value, dtype=np.float64)  # type: ignore
            return np_value.tobytes(order="C")

        return process
