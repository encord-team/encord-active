import sqlite3

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
    def __init__(self, project_file_structure: BaseProjectFileStructure) -> None:
        self.project_file_structure = project_file_structure

    def __enter__(self):
        self.conn = sqlite3.connect(self.project_file_structure.db)
        return self.conn

    def __exit__(self, type, value, traceback):
        self.conn.__exit__(type, value, traceback)


class PrismaConnection:
    def __init__(self, project_file_structure: BaseProjectFileStructure) -> None:
        self.datasource = DatasourceOverride(url=f"file:{project_file_structure}")

    def __enter__(self):
        self.db = Prisma(datasource=self.datasource)
        self.db.connect()
        return self.db

    def __exit__(self, type, value, traceback):
        if self.db.is_connected():
            self.db.disconnect()
