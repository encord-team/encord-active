import logging
import os
from typing import Optional, Callable

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import TypeDecorator, JSON
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.type_api import TypeEngine
from sqlalchemy.types import VARCHAR

__all__ = ["EncryptedAnnotationJSON"]


logger: logging.Logger = logging.getLogger("encrypted_annotation_json")


class EncryptedAnnotationJSON(TypeDecorator):
    """
    Special Wrapper around text type
    """

    impl = JSON
    cache_ok = True

    def __init__(self) -> None:
        super().__init__()
        key = os.environ.get("FERNET_SECRET", None)
        if key is None:
            self.fernet = None
        else:
            self.fernet = Fernet(key.encode("utf-8"))

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        return dialect.type_descriptor(JSON())  # type: ignore

    @staticmethod
    def _map_json_list(value: list[dict], callback: Callable[[bytes], bytes]) -> list[dict]:
        return [
            {
                k: v if k not in ["createdBy", "lastEditedBy"] else callback(v.encode("utf-8")).decode("utf-8")
                for k, v in elem.items()
            }
            for elem in value
        ]

    def process_bind_param(self, value: Optional[list[dict]], dialect: Dialect) -> Optional[list[dict]]:
        if value is None:
            return None
        if self.fernet is None:
            return value
        return self._map_json_list(value, lambda b: self.fernet.encrypt(b))

    def bind_processor(self, dialect: Dialect):
        def encrypt_json_list(value: Optional[str]) -> Optional[int]:
            return self.process_bind_param(value, dialect)

        return encrypt_json_list

    def result_processor(self, dialect: Dialect, coltype):
        def decrypt_json_list(value: Optional[list[dict]]) -> Optional[list[dict]]:
            if value is None:
                return None
            if self.fernet is None:
                return value
            try:
                return self._map_json_list(value, lambda b: self.fernet.encrypt(b))
            except InvalidToken as e:
                logging.error(f"Error decrypting email: {e}")
                return self._map_json_list(value, lambda b: b"")

        return decrypt_json_list
