from typing import Union, Optional

import numpy as np
from psycopg2.extensions import Binary
from sqlalchemy import Float, TypeDecorator
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.type_api import TypeEngine
from sqlalchemy.types import BINARY, UserDefinedType

__all__ = ["PGVector"]


class PGVectorRaw(UserDefinedType):
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super(UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kwargs) -> str:
        return f"VECTOR({self.dim})"

    class comparator_factory(UserDefinedType.Comparator):
        def l2_distance(self, other):
            return self.op("<->", return_type=Float)(other)

        def max_inner_product(self, other):
            return self.op("<#>", return_type=Float)(other)

        def cosine_distance(self, other):
            return self.op("<=>", return_type=Float)(other)


class PGVector(TypeDecorator):
    """Platform-independent PGVector type.

    Uses PostgreSQL's Vector type, otherwise uses
    JSON, storing as a serialized key.
    """

    impl = PGVectorRaw
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        self.dim = dim

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PGVectorRaw(self.dim))  # type: ignore
        else:
            return dialect.type_descriptor(BINARY())  # type: ignore

    def bind_processor(self, dialect: Dialect):
        def as_validated_ndarray(value: Union[np.ndarray, bytes, None]) -> Optional[np.ndarray]:
            if value is None:
                return None
            np_value = np.frombuffer(value, dtype=np.float64) if isinstance(value, bytes) else value
            if np_value.dtype != np.float64:
                raise ValueError(f"pgvector with wrong type: {np_value.dtype}")
            if np_value.ndim != 1 or np_value.shape != (self.dim,):
                raise ValueError(f"pgvector with wrong dimension: {np_value.shape}")
            return np_value

        if dialect.name == "postgresql":

            def process_pg(value: Union[np.ndarray, bytes, None]) -> Optional[str]:
                if value is None:
                    return None
                np_value = as_validated_ndarray(value)
                return "[" + ",".join([str(float(v)) for v in np_value]) + "]"

            return process_pg
        else:
            bind_other = BINARY().bind_processor(dialect)

            def process_fallback(value: Union[np.ndarray, bytes, None]) -> Optional[bytes]:
                if value is None:
                    return None
                np_value = as_validated_ndarray(value)
                return bind_other(np_value.tobytes(order="C"))  # type: ignore

            return process_fallback

    def result_processor(self, dialect: Dialect, coltype):
        result_processor = BINARY().result_processor(dialect, coltype)

        def process(value: Union[np.ndarray, str, bytes, Binary, None]) -> Optional[bytes]:
            if value is None:
                return None
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
