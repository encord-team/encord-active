from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Union

from prisma import models
from prisma.types import (
    DataUnitCreateInput,
    DataUnitInclude,
    DataUnitWhereInput,
    LabelRowCreateInput,
)

from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.file_structure.base import BaseProjectFileStructure


# To be deprecated when Encord Active version is >= 0.1.60.
def _fill_missing_tables(pfs: ProjectFileStructure):
    # Adds the content missing from the data units and label rows tables when projects with
    # older versions of Encord Active are handled with versions greater than 0.1.52.
    label_row_meta = json.loads(pfs.label_row_meta.read_text(encoding="utf-8"))
    with PrismaConnection(pfs) as conn:
        fill_label_rows = conn.labelrow.count() == 0
        fill_data_units = conn.dataunit.count() == 0
        if not (fill_label_rows or fill_data_units):
            return

        with conn.batch_() as batcher:
            for label_row in pfs.iter_labels():
                label_row_dict = json.loads(label_row.label_row_file.read_text(encoding="utf-8"))
                label_hash = label_row_dict["label_hash"]
                lr_data_hash = label_row_meta[label_hash]["data_hash"]
                data_type = label_row_dict["data_type"]

                if fill_label_rows:
                    label_hash = label_row_dict["label_hash"]
                    batcher.labelrow.create(
                        LabelRowCreateInput(
                            label_hash=label_hash,
                            data_hash=lr_data_hash,
                            data_title=label_row_dict["data_title"],
                            data_type=data_type,
                            created_at=label_row_meta[label_hash].get("created_at", datetime.now()),
                            last_edited_at=label_row_meta[label_hash].get("last_edited_at", datetime.now()),
                            location=label_row.label_row_file.as_posix(),
                        )
                    )

                if fill_data_units:
                    data_units = label_row_dict["data_units"]
                    for data_unit in label_row.iter_data_unit():
                        du = data_units[data_unit.hash]
                        if data_type == "video":
                            if "_" not in data_unit.path.stem:
                                continue
                            _, frame_str = data_unit.path.stem.rsplit("_", 1)
                        else:
                            frame_str = du.get("data_sequence", 0)
                        frame = int(frame_str)

                        batcher.dataunit.create(
                            DataUnitCreateInput(
                                data_hash=data_unit.hash,
                                data_title=du["data_title"],
                                frame=frame,
                                location=data_unit.path.as_posix(),
                                lr_data_hash=lr_data_hash,
                            )
                        )
            batcher.commit()


class DataUnitStructure(NamedTuple):
    hash: str
    path: Path


class LabelRowStructure:
    def __init__(self, path: Path, mappings: dict[str, str]):
        self.path: Path = path
        self._mappings: dict[str, str] = mappings
        self._rev_mappings: dict[str, str] = {v: k for k, v in mappings.items()}

    def __hash__(self) -> int:
        return hash(self.path.as_posix())

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
            old_du_hash, *_ = du_path.stem.split("_")
            new_du_hash = self._rev_mappings.get(old_du_hash, old_du_hash)
            yield DataUnitStructure(new_du_hash, du_path)

    def is_present(self):
        return self.path.is_dir()


class ProjectFileStructure(BaseProjectFileStructure):
    def __init__(self, project_dir: Union[str, Path]):
        super().__init__(project_dir)
        self._mappings = json.loads(self.mappings.read_text()) if self.mappings.exists() else {}

    def cache_clear(self) -> None:
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

    def data_units(
        self, where: Optional[DataUnitWhereInput] = None, include_label_row: bool = False
    ) -> List[models.DataUnit]:
        to_include = DataUnitInclude(label_row=True) if include_label_row else None
        with PrismaConnection(self) as conn:
            _fill_missing_tables(self)
            return conn.dataunit.find_many(where=where, include=to_include)

    def label_rows(self) -> List[models.LabelRow]:
        with PrismaConnection(self) as conn:
            _fill_missing_tables(self)
            return conn.labelrow.find_many()

    def iter_labels(self) -> Iterator[LabelRowStructure]:
        DATA_HASH_REGEX = r"([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})"
        pattern = re.compile(DATA_HASH_REGEX)
        for label_hash in self.data.iterdir():
            # Avoid unexpected folders in the data directory like `.DS_Store`
            if pattern.match(label_hash.name) is None:
                continue
            path = self.data / label_hash
            yield LabelRowStructure(path=path, mappings=self._mappings)

    @property
    def mappings(self) -> Path:
        return self.project_dir / "hash_mappings.json"

    def __repr__(self) -> str:
        return f"ProjectFileStructure({self.project_dir})"
