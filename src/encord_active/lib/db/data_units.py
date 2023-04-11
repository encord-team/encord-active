import json
from sqlite3 import OperationalError
from typing import Callable

from encord_active.lib.db.base import DataUnit
from encord_active.lib.db.connection import DBConnection

TABLE_NAME = "data_units"


def create_data_units_table():
    with DBConnection() as conn:
        conn.execute(
            f"""
             CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY,
                hash TEXT NOT NULL,
                location TEXT NOT NULL,
                title TEXT NOT NULL,
                frame INTEGER NOT NULL
             )
             """
        )


def ensure_existence(fn: Callable):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except OperationalError:
            create_data_units_table()

            # begin backwards compatibility code (read paths from old filesystem storage)
            project_file_structure = DBConnection.project_file_structure()
            data_units = []
            # fetch data units from local storage
            for label_hash in project_file_structure.data.iterdir():
                label_row_path = project_file_structure.data / label_hash / "label_row.json"
                label_row = json.loads(label_row_path.read_text(encoding="utf-8"))
                images_dir = project_file_structure.data / label_hash / "images"

                for data_unit in label_row["data_units"].values():
                    du_hash = data_unit["data_hash"]
                    du_title = data_unit["data_title"]

                    if label_row["data_type"] == "video":
                        for du_frame_path in images_dir.glob(f"{du_hash}_*"):
                            du_frame = int(du_frame_path.stem.rsplit("_", maxsplit=1)[-1])
                            data_units.append(
                                DataUnit(hash=du_hash, location=du_frame_path.resolve(), title=du_title, frame=du_frame)
                            )
                    else:
                        du_frame = int(data_unit["data_sequence"])
                        du_frame_path = next(images_dir.glob(f"{du_hash}.*"), None)
                        if du_frame_path is not None:
                            data_units.append(
                                DataUnit(hash=du_hash, location=du_frame_path.resolve(), title=du_title, frame=du_frame)
                            )

            # store data units references in the db
            DataUnits().create_many(data_units)
            # end backwards compatibility code

            return fn(*args, **kwargs)

    return wrapper


class DataUnits:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    @ensure_existence
    def all(self) -> list[DataUnit]:
        with DBConnection() as conn:
            return [
                DataUnit(du_hash, location, title, frame)
                for du_hash, location, title, frame in conn.execute(
                    f"SELECT hash, location, title, frame FROM {TABLE_NAME}"
                ).fetchall()
            ]

    @ensure_existence
    def get_row(self, du_hash: str) -> DataUnit:
        with DBConnection() as conn:
            row = conn.execute(
                f"SELECT hash, location, title, frame FROM {TABLE_NAME} where hash = {du_hash}"
            ).fetchone()
            if row is None:
                raise KeyError(f"There is no data unit with hash={du_hash}")
            return DataUnit(*row)

    @ensure_existence
    def create_many(self, data_units: list[DataUnit]):
        with DBConnection() as conn:
            return conn.executemany(
                f"INSERT INTO {TABLE_NAME} (hash, location, title, frame) VALUES(?, ?, ?, ?)", data_units
            )
