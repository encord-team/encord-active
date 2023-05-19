from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

from PIL import Image
from prisma import models
from prisma.types import (
    DataUnitCreateInput,
    DataUnitInclude,
    DataUnitWhereInput,
    LabelRowCreateInput,
)

from encord_active.lib.common.utils import download_file, download_image, extract_frames
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.encord.utils import get_encord_project
from encord_active.lib.file_structure.base import BaseProjectFileStructure
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.project.metadata import ProjectMeta, fetch_project_meta

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
                label_row_dict = label_row.label_row_json
                label_hash = label_row_dict["label_hash"]
                lr_data_hash = label_row_meta[label_hash]["data_hash"]
                data_type = label_row_dict["data_type"]

                if fill_label_rows:
                    label_hash = label_row_dict["label_hash"]
                    legacy_label_row_file = label_row.label_row_file_deprecated_for_migration().read_text("utf-8")
                    batcher.labelrow.create(
                        LabelRowCreateInput(
                            label_hash=label_hash,
                            data_hash=lr_data_hash,
                            data_title=label_row_dict["data_title"],
                            data_type=data_type,
                            created_at=label_row_meta[label_hash].get("created_at", datetime.now()),
                            last_edited_at=label_row_meta[label_hash].get("last_edited_at", datetime.now()),
                            location=None,
                            label_row_json=legacy_label_row_file,
                        )
                    )

                if fill_data_units:
                    data_units = label_row_dict["data_units"]
                    for data_unit in label_row.iter_data_unit():
                        legacy_lr_path = label_row.label_row_file_deprecated_for_migration().parent / "data"
                        if data_unit.frame is not None:
                            legacy_du_path = next(legacy_lr_path.glob(f"{data_unit.du_hash}_{data_unit.frame}.*"))
                        else:
                            legacy_du_path = next(legacy_lr_path.glob(f"{data_unit.du_hash}.*"))

                        du = data_units[data_unit.du_hash]
                        if data_type == "video":
                            if "_" not in legacy_du_path.stem:
                                frame_str = "-1"  # To include a reference to the video location in the DataUnit table
                            else:
                                _, frame_str = legacy_du_path.stem.rsplit("_", 1)
                        else:
                            frame_str = du.get("data_sequence", 0)
                        frame = int(frame_str)

                        batcher.dataunit.create(
                            DataUnitCreateInput(
                                data_hash=data_unit.du_hash,
                                data_title=du["data_title"],
                                frame=frame,
                                lr_data_hash=lr_data_hash,
                                data_link=legacy_du_path.as_uri(),
                            )
                        )
            batcher.commit()


class DataUnitStructure(NamedTuple):
    label_hash: str
    du_hash: str
    data_type: str
    # If type is 'video' this is the video link not the image link
    signed_url: str
    # Video frame or sequence index
    frame: int


class LabelRowStructure:
    def __init__(self, mappings: dict[str, str], label_hash: str, project: "ProjectFileStructure"):
        self._mappings: dict[str, str] = mappings
        self._rev_mappings: dict[str, str] = {v: k for k, v in mappings.items()}
        self._label_hash = label_hash
        self._project = project

    def __hash__(self) -> int:
        return hash(self._label_hash)

    def label_row_file_deprecated_for_migration(self) -> Path:
        return self._project.project_dir / "data" / self._label_hash

    @property
    def label_row_json(self) -> Dict[str, Any]:
        with PrismaConnection(self._project) as conn:
            entry = conn.labelrow.find_unique(where={"label_hash": self._label_hash})
            return json.loads(entry.label_row_json)

    def set_label_row_json(self, label_row_json: Dict[str, Any]) -> None:
        with PrismaConnection(self._project) as conn:
            conn.labelrow.update(
                where={"label_hash": self._label_hash}, data={"label_row_json": json.dumps(label_row_json, ident=2)}
            )

    def clone_label_row_json(self, source: "LabelRowStructure", label_row_json: str, local_path: Optional[str]) -> None:
        with PrismaConnection(source._project) as conn:
            rows = conn.labelrow.find_many(
                where={
                    "label_hash": self._label_hash,
                },
                include={
                    "data_units": True,
                },
            )
        with PrismaConnection(self._project) as conn:
            for row in rows:
                conn.labelrow.create(
                    {
                        **row,
                        "label_row_json": label_row_json,
                        "local_path": local_path,
                    },
                    include={
                        "data_units": True,
                    },
                )

    def iter_data_unit(
        self, data_unit_hash: Optional[str] = None, frame: Optional[int] = None
    ) -> Iterator[DataUnitStructure]:
        with PrismaConnection(self._project) as conn:
            where = {"label_hash": {"equals": self._label_hash}}
            if data_unit_hash is not None:
                du_hash = self._mappings.get(data_unit_hash, data_unit_hash)
                where["data_hash"] = {"equals": du_hash}
            all_rows = conn.labelrow.find_many(
                where=where,
                include={
                    "data_units": True
                    if frame is None
                    else {
                        "where": {
                            "frame": {
                                "equals": frame,
                            }
                        }
                    }
                },
            )
            encord_project_metadata = self._project.load_project_meta()
            encord_project = get_encord_project(
                encord_project_metadata["ssh_key_path"], encord_project_metadata["project_hash"]
            )
            cached_signed_urls = self._project.cached_signed_urls
            for label_row in all_rows:
                data_links = {}
                if any(data_unit.data_hash not in cached_signed_urls for data_unit in label_row.data_units):
                    data_links = encord_project.get_label_row(
                        label_row.label_hash,
                        get_signed_url=True,
                    )
                for data_unit in label_row.data_units:
                    du_hash = data_unit.data_hash
                    new_du_hash = self._rev_mappings.get(du_hash, du_hash)
                    if data_unit.data_hash in cached_signed_urls:
                        signed_url = cached_signed_urls[data_unit.data_hash]
                        data_type = "image"  # FIXME: this is wrong, store in prisma db
                    else:
                        signed_url = data_links["data_units"][du_hash]["data_link"]
                        data_type = data_links["data_units"][du_hash]["data_type"]
                        cached_signed_urls[data_unit.data_hash] = signed_url
                    yield DataUnitStructure(label_row.label_hash, new_du_hash, data_type, signed_url, data_unit.frame)

    def iter_data_unit_with_image(
        self, data_unit_hash: Optional[str] = None, frame: Optional[int] = None
    ) -> Iterator[Tuple[DataUnitStructure, Image.Image]]:
        # Temporary directory for all video decodes
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for data_unit_struct in self.iter_data_unit(data_unit_hash=data_unit_hash, frame=frame):
                if data_unit_struct.data_type in {"img_group", "image"}:
                    image = download_image(data_unit_struct.signed_url)
                    yield data_unit_struct, image
                elif data_unit_struct.data_type == "video":
                    video_dir = tmp_path / data_unit_struct.du_hash
                    images_dir = video_dir / "frames"
                    existing_image = next(
                        images_dir.glob(f"{data_unit_struct.du_hash}_{data_unit_struct.frame}.*"), None
                    )
                    if existing_image is not None:
                        return data_unit_struct, Image.open(existing_image)
                    else:
                        video_file = video_dir / "video"
                        download_file(data_unit_struct.signed_url, video_file)
                        extract_frames(video_file, images_dir, data_unit_struct.du_hash)
                    downloaded_image = next(images_dir.glob(f"{data_unit_struct.du_hash}_{data_unit_struct.frame}.*"))
                    return data_unit_struct, Image.open(downloaded_image)
                else:
                    raise RuntimeError("Unsupported data type")


class ProjectFileStructure(BaseProjectFileStructure):
    def __init__(self, project_dir: Union[str, Path]):
        super().__init__(project_dir)
        self._mappings = json.loads(self.mappings.read_text()) if self.mappings.exists() else {}
        self._cached_project_meta: "Optional[ProjectMeta]" = None
        self.cached_signed_urls: Dict[str, str] = {}

    def cache_clear(self) -> None:
        self._mappings = json.loads(self.mappings.read_text()) if self.mappings.exists() else {}
        if self._cached_project_meta is not None:
            self._cached_project_meta = fetch_project_meta(self.project_dir)

    def load_project_meta(self) -> ProjectMeta:
        if self._cached_project_meta is not None:
            return self._cached_project_meta
        else:
            cache = fetch_project_meta(self.project_dir)
            self._cached_project_meta = cache
            return cache

    @property
    def data_legacy_folder(self) -> Path:
        return self.project_dir / "data"

    @property
    def local_data_store(self) -> Path:
        return self.project_dir / "local_data"

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
        return LabelRowStructure(mappings=self._mappings, label_hash=label_hash, project=self)

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
            if label_hash is None:
                continue
            yield LabelRowStructure(mappings=self._mappings, label_hash=label_hash, project=self)

    @property
    def mappings(self) -> Path:
        return self.project_dir / "hash_mappings.json"

    def __repr__(self) -> str:
        return f"ProjectFileStructure({self.project_dir})"
