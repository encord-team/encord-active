import logging
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import TypeDecorator
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.type_api import TypeEngine
from sqlalchemy.types import VARCHAR

__all__ = ["EncryptedStr"]


logger: logging.Logger = logging.getLogger("encrypted_str")


class EncryptedStr(TypeDecorator):
    """
    Special Wrapper around text type
    """

    impl = VARCHAR
    cache_ok = True

    def __init__(self) -> None:
        super().__init__()
        key = os.environ.get("FERNET_SECRET", None)
        if key is None:
            self.fernet = None
        else:
            self.fernet = Fernet(key.encode("utf-8"))

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        return dialect.type_descriptor(VARCHAR(600))  # type: ignore

    def process_bind_param(self, value: Optional[str], dialect: Dialect) -> Optional[str]:
        if value is None:
            return None
        if self.fernet is None:
            return value
        return self.fernet.encrypt(value.encode("utf-8")).decode("utf-8")

    def bind_processor(self, dialect: Dialect):
        def encrypt_str(value: Optional[str]) -> Optional[str]:
            return self.process_bind_param(value, dialect)

        return encrypt_str

    def result_processor(self, dialect: Dialect, coltype):
        def decrypt_str(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            if self.fernet is None:
                return value
            try:
                return self.fernet.decrypt(value.encode("utf-8")).decode("utf-8")
            except InvalidToken as e:
                logging.error(f"Error decrypting email: {e}")
                return ""

        return decrypt_str