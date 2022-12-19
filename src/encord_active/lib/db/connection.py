import sqlite3
from pathlib import Path


class DBConnection:
    file = None

    def __enter__(self):
        if not self.file:
            raise ConnectionError("DB file was not set, call `DBConnection.set_dbfile(<some/file>)`")

        self.conn = sqlite3.connect(self.file)
        return self.conn

    def __exit__(self, type, value, traceback):
        self.conn.__exit__(type, value, traceback)

    @classmethod
    def set_dbfile(cls, file: Path):
        cls.file = file
