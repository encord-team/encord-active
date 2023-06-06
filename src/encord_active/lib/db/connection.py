import sqlite3
import typing
from typing import Optional

if typing.TYPE_CHECKING:
    import prisma

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


# FIXME: _PRISMA_DB_GLOBAL_CACHE: typing.Dict[str, "prisma.Prisma"] = {}


class PrismaConnection:
    def __init__(
        self,
        project_file_structure: BaseProjectFileStructure,
        cache_db: Optional["prisma.Prisma"] = None,
        unsafe_force: bool = False,
    ) -> None:
        ensure_prisma_db(project_file_structure.prisma_db)
        from prisma.types import DatasourceOverride

        self.cache_db = cache_db
        self.db: "Optional[prisma.Prisma]" = None
        self.unsafe_force = unsafe_force
        self.datasource = DatasourceOverride(url=f"file:{project_file_structure.prisma_db.absolute()}")

    def __enter__(self) -> "prisma.Prisma":
        from prisma import Prisma

        if self.cache_db is not None:
            return self.cache_db

        # cache_key = self.datasource["url"]
        # if cache_key in _PRISMA_DB_GLOBAL_CACHE:
        #    if self.unsafe_force:
        #        db = _PRISMA_DB_GLOBAL_CACHE[cache_key]
        #        if db.is_connected():
        #            db.disconnect()
        #    else:
        #        return _PRISMA_DB_GLOBAL_CACHE[cache_key]

        self.db = Prisma(datasource=self.datasource)
        self.db.connect()
        # _PRISMA_DB_GLOBAL_CACHE[cache_key] = db  # Never disconnect, global prisma connection
        return self.db

    def __exit__(self, type, value, traceback):
        if self.db is not None and self.db.is_connected():
            self.db.disconnect()
            self.db = None
