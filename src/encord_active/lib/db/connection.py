import sqlite3
from typing import Optional

from encord_active.lib.db.prisma_init import generate_prisma_client

try:
    import prisma
    import prisma.client
except (RuntimeError, ImportError):
    generate_prisma_client()
    from importlib import reload

    reload(prisma)  # pylint: disable=used-before-assignment
finally:
    from prisma import Prisma
    from prisma.types import DatasourceOverride


# uses prisma so must appear after prisma schema generation
from encord_active.lib.file_structure.base import BaseProjectFileStructure


class DBConnection:
    _project_file_structure: Optional[BaseProjectFileStructure] = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.project_file_structure().db)
        return self.conn

    def __exit__(self, type, value, traceback):
        self.conn.__exit__(type, value, traceback)

    @classmethod
    def set_project_file_structure(cls, project_file_structure: BaseProjectFileStructure):
        cls._project_file_structure = project_file_structure

    @classmethod
    def project_file_structure(cls):
        if not cls._project_file_structure:
            raise ConnectionError(
                "`project_file_structure` is not set, call `DBConnection.set_project_file_structure(..)` first"
            )
        return cls._project_file_structure


class PrismaConnection:
    _datasource: Optional[DatasourceOverride] = None
    _project_file_structure: Optional[BaseProjectFileStructure] = None

    def __enter__(self):
        self.db = Prisma(datasource=self.datasource())
        self.db.connect()
        return self.db

    def __exit__(self, type, value, traceback):
        if self.db.is_connected():
            self.db.disconnect()

    @classmethod
    def set_project_file_structure(cls, project_file_structure: BaseProjectFileStructure):
        if cls._project_file_structure is project_file_structure:  # skip if it was already set
            return

        cls._project_file_structure = project_file_structure
        url = f"file:{cls._project_file_structure.prisma_db}"
        cls._datasource = DatasourceOverride(url=url)

    @classmethod
    def project_file_structure(cls):
        if not cls._project_file_structure:
            raise ConnectionError(
                "`project_file_structure` is not set, call `PrismaConnection.set_project_file_structure(..)` first"
            )
        return cls._project_file_structure

    @classmethod
    def datasource(cls) -> DatasourceOverride:
        if not cls._datasource:
            raise ConnectionError(
                "`project_file_structure` is not set, call `PrismaConnection.set_project_file_structure(..)` first"
            )
        return cls._datasource
