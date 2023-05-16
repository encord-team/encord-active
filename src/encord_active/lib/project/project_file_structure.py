from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Union, Dict, Any

from prisma import models
from prisma.types import (
    DataUnitCreateInput,
    DataUnitInclude,
    DataUnitWhereInput,
    LabelRowCreateInput,
)

from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.file_structure.base import BaseProjectFileStructure
from encord_active.lib.metrics.types import EmbeddingType

EMBEDDING_TYPE_TO_FILENAME = {
    EmbeddingType.IMAGE: "cnn_images.pkl",
    EmbeddingType.CLASSIFICATION: "cnn_classifications.pkl",
    EmbeddingType.OBJECT: "cnn_objects.pkl",
}

EMBEDDING_REDUCED_TO_FILENAME = {
    EmbeddingType.IMAGE: "cnn_images_reduced.pkl",
    EmbeddingType.CLASSIFICATION: "cnn_classifications_reduced.pkl",
    EmbeddingType.OBJECT: "cnn_objects_reduced.pkl",
}


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
                                frame_str = "-1"  # To include a reference to the video location in the DataUnit table
                            else:
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


class LabelRowStructure:
    def __init__(self, mappings: dict[str, str], label_hash: str, project: "ProjectFileStructure"):
        self._mappings: dict[str, str] = mappings
        self._rev_mappings: dict[str, str] = {v: k for k, v in mappings.items()}
        self._label_hash = label_hash
        self._project = project

    def __hash__(self) -> int:
        return hash(self._label_hash)

    @property
    def label_row_json(self) -> Dict[str, Any]:
        with PrismaConnection(self._project) as conn:
            entry = conn.labelrow.find_unique(
                where={}
            )
            return json.loads(entry.label_row_json)

    def iter_data_unit(
        self, data_unit_hash: Optional[str] = None, frame: Optional[int] = None
    ) -> Iterator[DataUnitStructure]:
        with PrismaConnection(self._project) as conn:
            where = {"label_hash": {"equals": self._label_hash}}
            data_unit_hash = self._mappings.get(data_unit_hash, data_unit_hash)
            if data_unit_hash is None:
                where["data_hash"] = {"equals": data_unit_hash}
            all_rows = conn.labelrow.find_many(where=where)
            where = {
                "label_row": {
                    "in": all_rows,
                }
            }
            if frame is not None:
                where["frame"] = {
                    "equals": [frame]
                }
            all_data_units = conn.dataunit.find_many(
                where=where
            )
            for data_unit in all_data_units:
                du_hash = data_unit.data_hash
                new_du_hash = self._rev_mappings.get(du_hash, du_hash)
                yield DataUnitStructure(new_du_hash)


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

    def get_embeddings_file(self, type_: EmbeddingType, reduced: bool = False) -> Path:
        lookup = EMBEDDING_REDUCED_TO_FILENAME if reduced else EMBEDDING_TYPE_TO_FILENAME
        return self.embeddings / lookup[type_]

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
        label_hash = self._mappings.get(label_hash, label_hash)
        path = self.data / label_hash
        return LabelRowStructure(path=path, mappings=self._mappings, label_hash=label_hash, project=self)

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
        label_rows = self.label_rows()
        for label_row in label_rows:
            label_hash = label_row.label_hash
            path = self.data / label_hash
            yield LabelRowStructure(path=path, mappings=self._mappings, label_hash=label_hash, project=self)

    @property
    def mappings(self) -> Path:
        return self.project_dir / "hash_mappings.json"

    def __repr__(self) -> str:
        return f"ProjectFileStructure({self.project_dir})"
