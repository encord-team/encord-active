import json
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Union

from prisma.models import DataUnit

from encord_active.lib.db.base import DataUnitLike
from encord_active.lib.db.compatibility import fill_data_units_table
from encord_active.lib.db.connection import DBConnection, PrismaConnection
from encord_active.lib.file_structure.base import BaseProjectFileStructure


class DataUnitStructure(NamedTuple):
    hash: str
    path: Path


class LabelRowStructure:
    def __init__(self, path: Path, mappings: dict[str, str]):
        self.path: Path = path
        self._mappings: dict[str, str] = mappings
        self._rev_mappings: dict[str, str] = {v: k for k, v in mappings.items()}

    @property
    def label_row_file(self) -> Path:
        return self.path / "label_row.json"

    @property
    def images_dir(self) -> Path:
        return self.path / "images"

    def iter_data_unit(
        self, data_unit_hash: Optional[str] = None, frame: Optional[int] = None
    ) -> Iterator[DataUnitStructure]:
        if data_unit_hash:
            glob_string = self._mappings.get(data_unit_hash, data_unit_hash)
        else:
            glob_string = "*"
        if frame:
            glob_string += f"_{frame}"
        glob_string += ".*"
        for du_path in self.images_dir.glob(glob_string):
            du_hash = du_path.name.split(".")[0].split("_")[0]
            yield DataUnitStructure(self._rev_mappings.get(du_hash, du_hash), du_path)

    def is_present(self):
        return self.path.is_dir()


class ProjectFileStructure(BaseProjectFileStructure):
    def __init__(self, project_dir: Union[str, Path]):
        super().__init__(project_dir)
        DBConnection.set_project_file_structure(self)
        self._mappings = json.loads(self.mappings.read_text()) if self.mappings.exists() else {}

    @property
    def data(self) -> Path:
        return self.project_dir / "data"

    @property
    def metrics(self) -> Path:
        return self.project_dir / "metrics"

    @property
    def metrics_meta(self) -> Path:
        return self.metrics / "metrics_meta.json"

    @property
    def embeddings(self) -> Path:
        return self.project_dir / "embeddings"

    @property
    def predictions(self) -> Path:
        return self.project_dir / "predictions"

    @property
    def db(self) -> Path:
        return self.project_dir / "sqlite.db"

    @property
    def prisma_db(self) -> Path:
        return self.project_dir / "prisma.db"

    @property
    def label_row_meta(self) -> Path:
        return self.project_dir / "label_row_meta.json"

    @property
    def image_data_unit(self) -> Path:
        return self.project_dir / "image_data_unit.json"

    @property
    def ontology(self) -> Path:
        return self.project_dir / "ontology.json"

    @property
    def project_meta(self) -> Path:
        return self.project_dir / "project_meta.yaml"

    def label_row_structure(self, label_hash: str) -> LabelRowStructure:
        path = self.data / self._mappings.get(label_hash, label_hash)
        return LabelRowStructure(path=path, mappings=self._mappings)

    def data_units(self, pattern: Optional[DataUnitLike] = None) -> Iterator[DataUnit]:
        with PrismaConnection() as conn:
            # add backwards compatibility code. Remove when Encord Active version is >= 0.1.60.
            if conn.dataunit.count() == 0:
                fill_data_units_table()
            # end backwards compatibility code.
            if pattern is None:
                return iter(conn.dataunit.find_many())
            else:
                and_clause = []
                for field in pattern._fields:
                    if (value := getattr(pattern, field)) is not None:
                        and_clause.append({field: value})
                return iter(conn.dataunit.find_many(where={"AND": and_clause}))

    def label_rows(self) -> Iterator:
        # use internally iter_labels() while the label row table does not exist in the db
        return self.iter_labels()

    def iter_labels(self) -> Iterator[LabelRowStructure]:
        for label_hash in self.data.iterdir():
            path = self.data / label_hash
            yield LabelRowStructure(path=path, mappings=self._mappings)

    @property
    def mappings(self) -> Path:
        return self.project_dir / "hash_mappings.json"
