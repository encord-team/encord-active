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
        ensure_prisma_db(project_file_structure.prisma_db)
        from prisma.types import DatasourceOverride
        self.pfs = project_file_structure
        self.datasource = DatasourceOverride(url=f"file:{project_file_structure.prisma_db}")

    def __enter__(self):
        from prisma import Prisma
        if self.pfs.prisma_db_conn_cache is not None:
            self.pfs.prisma_db_conn_cache_counter += 1
            return self.pfs.prisma_db_conn_cache

        db = Prisma(datasource=self.datasource)
        db.connect()
        self.pfs.prisma_db_conn_cache = db
        self.pfs.prisma_db_conn_cache_counter = 1
        return db

    def __exit__(self, type, value, traceback):
        self.pfs.prisma_db_conn_cache_counter -= 1
        if self.pfs.prisma_db_conn_cache_counter == 0:
            db = self.pfs.prisma_db_conn_cache
            self.pfs.prisma_db_conn_cache = None
            if db.is_connected():
                db.disconnect()
