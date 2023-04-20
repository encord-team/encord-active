import sqlite3
from pathlib import Path
from typing import Optional

from prisma import Prisma
from prisma.cli.prisma import run
from prisma.types import DatasourceOverride

from encord_active.lib.project import ProjectFileStructure

PRISMA_SCHEMA_FILE = Path(__file__).parent / "prisma.schema"


class DBConnection:
    _project_file_structure: Optional[ProjectFileStructure] = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.project_file_structure().db)
        return self.conn

    def __exit__(self, type, value, traceback):
        self.conn.__exit__(type, value, traceback)

    @classmethod
    def set_project_path(cls, project_path: Path):
        cls._project_file_structure = ProjectFileStructure(project_path)

    @classmethod
    def project_file_structure(cls):
        if not cls._project_file_structure:
            raise ConnectionError("`project_path` was not set, call `DBConnection.set_project_path('path/to/project')`")
        return cls._project_file_structure


class PrismaConnection:
    _datasource: Optional[DatasourceOverride] = None

    def __enter__(self):
        self.db = Prisma(datasource=self.datasource())
        self.db.connect()
        return self.db

    def __exit__(self, type, value, traceback):
        if self.db.is_connected():
            self.db.disconnect()

    @classmethod
    def set_project_path(cls, project_path: Path):
        db_file = ProjectFileStructure(project_path).prisma_db
        url = f"file:{db_file}"
        env = {"MY_DATABASE_URL": url}

        run(["db", "push", f"--schema={PRISMA_SCHEMA_FILE}"], env=env)
        cls._datasource = DatasourceOverride(url=url)

    @classmethod
    def datasource(cls) -> DatasourceOverride:
        if not cls._datasource:
            raise ConnectionError(
                "`project_path` was not set, call `PrismaConnection.set_project_path('path/to/project')`"
            )
        return cls._datasource
