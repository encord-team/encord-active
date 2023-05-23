import sqlite3

import encord_active.lib.db  # pylint: disable=unused-import
from encord_active.lib.db.prisma_init import ensure_prisma_db
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
    def __init__(self, project_file_structure: BaseProjectFileStructure) -> None:
        if not project_file_structure.prisma_db.exists():
            ensure_prisma_db(project_file_structure.prisma_db)
        from prisma.types import DatasourceOverride

        self.datasource = DatasourceOverride(url=f"file:{project_file_structure.prisma_db}")

    def __enter__(self):
        from prisma import Prisma

        self.db = Prisma(datasource=self.datasource)
        self.db.connect()
        return self.db

    def __exit__(self, type, value, traceback):
        if self.db.is_connected():
            self.db.disconnect()
