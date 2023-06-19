import sqlite3
import typing
from typing import Optional

if typing.TYPE_CHECKING:
    import prisma

import encord_active.lib.db  # pylint: disable=unused-import
from encord_active.lib.file_structure.base import BaseProjectFileStructure


class DBConnection:
    def __init__(self, project_file_structure: BaseProjectFileStructure) -> None:
        self.project_file_structure = project_file_structure

    def __enter__(self):
        self.conn = sqlite3.connect(self.project_file_structure.db)
        return self.conn

    def __exit__(self, type, value, traceback):
        self.conn.__exit__(type, value, traceback)


class PrismaConnection:
    def __init__(
        self,
        project_file_structure: BaseProjectFileStructure,
        cache_db: Optional["prisma.Prisma"] = None,
    ) -> None:
        from prisma.types import DatasourceOverride

        self.cache_db = cache_db
        self.db: "Optional[prisma.Prisma]" = None
        self.datasource = DatasourceOverride(url=f"file:{project_file_structure.prisma_db.absolute()}")

    def __enter__(self) -> "prisma.Prisma":
        from prisma import Prisma

        if self.cache_db is not None:
            return self.cache_db

        self.db = Prisma(datasource=self.datasource)
        self.db.connect()
        return self.db

    def __exit__(self, type, value, traceback):
        if self.db is not None and self.db.is_connected():
            self.db.disconnect()
            self.db = None
