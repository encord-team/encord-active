from datetime import datetime
from typing import List, Optional, Type
from uuid import UUID

from sqlalchemy import (
    INT,
    JSON,
    REAL,
    SMALLINT,
    TIMESTAMP,
    BigInteger,
    Boolean,
    Column,
    Integer,
    LargeBinary,
)
from sqlmodel import CheckConstraint, Field, ForeignKeyConstraint, SQLModel
from sqlmodel.sql.sqltypes import GUID, AutoString

from encord_active.analysis.types import EmbeddingDistanceMetric
from encord_active.db.enums import (
    AnnotationType,
    AnnotationTypeMaxValue,
    DataType,
    DataTypeMaxValue,
)
from encord_active.db.util.char8 import Char8
from encord_active.db.util.encrypted_annotation_json import EncryptedAnnotationJSON
from encord_active.db.util.encrypted_str import EncryptedStr


def fk_constraint(
    fields: List[str],
    table: Type[SQLModel],
    name: str,
    foreign_fields: Optional[List[str]] = None,
) -> ForeignKeyConstraint:
    return ForeignKeyConstraint(
        fields,
        [getattr(table, field) for field in foreign_fields or fields],
        onupdate="CASCADE",
        ondelete="CASCADE",
        name=name,
    )


def field_uuid(primary_key: bool = False, unique: bool = False, nullable: bool = False) -> UUID:
    return Field(
        primary_key=primary_key, unique=unique, sa_column=Column(GUID(), nullable=nullable, primary_key=primary_key)
    )


def field_bool(nullable: bool = False) -> bool:
    return Field(sa_column=Column(Boolean, nullable=nullable))


def field_int(primary_key: bool = False, nullable: bool = False) -> int:
    return Field(primary_key=primary_key, sa_column=Column(Integer, nullable=nullable, primary_key=primary_key))


def field_small_int(primary_key: bool = False, nullable: bool = False) -> int:
    return Field(primary_key=primary_key, sa_column=Column(SMALLINT, nullable=nullable, primary_key=primary_key))


def field_big_int(primary_key: bool = False, nullable: bool = False) -> int:
    return Field(primary_key=primary_key, sa_column=Column(BigInteger, nullable=nullable, primary_key=primary_key))


def field_datetime(nullable: bool = False) -> datetime:
    return Field(sa_column=Column(TIMESTAMP, nullable=nullable))


def field_real(nullable: bool = False) -> float:
    return Field(sa_column=Column(REAL, nullable=nullable))


def field_text(nullable: bool = False) -> str:
    return Field(sa_column=Column(AutoString, nullable=nullable))


def field_encrypted_text(nullable: bool = False) -> str:
    return Field(sa_column=Column(EncryptedStr, nullable=nullable))


def field_bytes(nullable: bool = False) -> bytes:
    return Field(sa_column=Column(LargeBinary, nullable=nullable))


def field_json_dict(nullable: bool = False, default: Optional[dict] = None) -> dict:
    return Field(sa_column=Column(JSON, nullable=nullable, default=default))


def field_json_list(nullable: bool = False, default: Optional[list] = None) -> list:
    return Field(sa_column=Column(JSON, nullable=nullable, default=default))


def field_annotation_json_list(nullable: bool = False, default: Optional[List[dict]] = None) -> List[dict]:
    return Field(sa_column=Column(EncryptedAnnotationJSON, nullable=nullable, default=default))


EmbeddingVector = bytes  # FIXME: Union[list[float], bytes]


def field_embedding_vector(dim: int, nullable: bool = False) -> EmbeddingVector:
    return Field(sa_column=Column(LargeBinary(dim), nullable=nullable))


def field_string_dict(nullable: bool = False) -> dict[str, str]:
    return Field(sa_column=Column(JSON(), nullable=nullable))


def field_metric_normal(nullable: bool = True) -> float:
    return Field(
        sa_column=Column(REAL, nullable=nullable),
    )


def field_metric_positive_integer(nullable: bool = True) -> int:
    return Field(sa_column=Column(INT, nullable=nullable))


def field_metric_positive_float(nullable: bool = True) -> float:
    return Field(sa_column=Column(REAL, nullable=nullable))


def field_char8(primary_key: bool = False, nullable: bool = False) -> str:
    return Field(
        primary_key=primary_key,
        sa_column=Column(Char8, nullable=nullable, primary_key=primary_key),
        min_length=8,
        max_length=8,
    )


def field_annotation_type() -> AnnotationType:
    return Field(sa_column=Column(SMALLINT, nullable=False))


def check_annotation_type(name: str) -> CheckConstraint:
    return CheckConstraint(f"annotation_type >= 0 AND annotation_type <= {AnnotationTypeMaxValue}", name=name)


def field_data_type() -> DataType:
    return Field(sa_column=Column(SMALLINT), nullable=False)


def check_data_type(name: str) -> CheckConstraint:
    return CheckConstraint(f"data_type >= 0 AND data_type <= {DataTypeMaxValue}", name=name)


def field_embedding_distance_metric(primary_key: bool = False) -> EmbeddingDistanceMetric:
    return Field(sa_column=Column(SMALLINT, nullable=False, primary_key=primary_key))
