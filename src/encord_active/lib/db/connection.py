import sqlite3
from pathlib import Path
from typing import Optional

from encord_active.lib.project import ProjectFileStructure


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
