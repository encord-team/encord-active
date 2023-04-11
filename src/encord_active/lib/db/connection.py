import sqlite3
from pathlib import Path
from typing import Optional

from encord_active.lib.file_structure.base import BaseProjectFileStructure


class DBConnection:
    _project_file_structure: Optional[BaseProjectFileStructure] = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.project_file_structure().db)
        return self.conn

    def __exit__(self, type, value, traceback):
        self.conn.__exit__(type, value, traceback)

    @classmethod
    def set_project_path(cls, project_path: Path):
        pass

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
