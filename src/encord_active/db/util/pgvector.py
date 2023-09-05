from typing import Union

import numpy as np
from sqlalchemy import TypeDecorator
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.type_api import TypeEngine
from sqlalchemy.types import UserDefinedType, BINARY

__all__ = ['PGVector']


class PGVectorRaw(UserDefinedType):
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super(UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kwargs) -> str:
        return f"VECTOR({self.dim})"


class PGVector(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.

    """

    impl = PGVectorRaw
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        self.dim = dim

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PGVectorRaw(self.dim))
        else:
            return dialect.type_descriptor(BINARY())

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
                return '[' + ','.join([str(float(v)) for v in np_value]) + ']'

            return process_pg
        else:
            bind_other = BINARY().bind_processor(dialect)

            def process_fallback(value: Union[np.ndarray, bytes]) -> bytes:
                np_value = as_validated_ndarray(value)
                return bind_other(np_value.tobytes(order='C'))

            return process_fallback

    def result_processor(self, dialect: Dialect, coltype):
        result_processor = BINARY().bind_processor(dialect)

        def process(value: Union[np.ndarray, str, bytes]) -> bytes:
            if isinstance(value, str):
                np_value = np.array(value[1:-1].split(','), dtype=np.float64)
            elif isinstance(value, np.ndarray):
                np_value = value
            else:
                np_value = np.frombuffer(result_processor(value), dtype=np.float64)
            return np_value.tobytes(order='C')
        return process
