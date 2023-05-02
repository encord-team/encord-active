import sqlite3
from pathlib import Path
from typing import Optional

from prisma.cli.prisma import run

from encord_active.lib.common.decorators import silence_stdout

PRISMA_SCHEMA_FILE = Path(__file__).parent / "schema.prisma"
run = silence_stdout(run)

try:
    import prisma
    from prisma import Prisma
    from prisma.models import DataUnit, LabelRow  # pylint: disable=unused-import
    from prisma.types import DatasourceOverride
except (RuntimeError, ImportError):
    run(["generate", f"--schema={PRISMA_SCHEMA_FILE}"])

    from importlib import reload

    reload(prisma)  # pylint: disable=used-before-assignment
    reload(prisma.models)
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
        db_file = cls._project_file_structure.prisma_db
        url = f"file:{db_file}"
        env = {"MY_DATABASE_URL": url}

        run(["db", "push", f"--schema={PRISMA_SCHEMA_FILE}", "--skip-generate"], env=env)
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
