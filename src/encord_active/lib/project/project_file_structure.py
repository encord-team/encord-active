from __future__ import annotations

import functools
import json
import tempfile
import typing
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import encord
from PIL import Image

if typing.TYPE_CHECKING:
    import prisma

from encord_active.lib.common.data_utils import (
    download_file,
    download_image,
    extract_frames,
)
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


class DataUnitStructure(NamedTuple):
    label_hash: str
    du_hash: str
    data_type: str
    # If type is 'video' this is the video link not the image link
    signed_url: str
    # Video frame or sequence index
    frame: Optional[int]
    # Raw data_hash
    data_hash_raw: str
    # Data unit metadata
    width: int
    height: int
    frames_per_second: float


@functools.lru_cache(maxsize=10)
def get_encord_project_cached(ssh_key_path: str, project_hash: str) -> encord.Project:
    return get_encord_project(ssh_key_path=ssh_key_path, project_hash=project_hash)


class LabelRowStructure:
    def __init__(
        self,
        mappings: dict[str, str],
        label_hash: str,
        project: "ProjectFileStructure",
        label_row: Optional["prisma.models.LabelRow"],
    ):
        self._mappings: dict[str, str] = mappings
        self._rev_mappings: dict[str, str] = {v: k for k, v in mappings.items()}
        self._label_hash = label_hash
        self._project = project
        self._label_row = label_row

    def __hash__(self) -> int:
        return hash(self._label_hash)

    @property
    def project(self) -> "ProjectFileStructure":
        return self._project

    def label_row_file_deprecated_for_migration(self) -> Path:
        return self._project.project_dir / "data" / self._label_hash / "label_row.json"

    def get_label_row_from_db(self, cache_db: Optional[prisma.Prisma] = None) -> "prisma.models.LabelRow":
        if self._label_row is not None:
            return self._label_row
        with PrismaConnection(self._project, cache_db=cache_db) as conn:
            res = conn.labelrow.find_unique(where={"label_hash": self._label_hash})
            if res is None:
                raise ValueError(f"label_row missing in prisma db(label_hash={self._label_hash})")
            return res

    @property
    def label_row_from_db(self) -> "prisma.models.LabelRow":
        return self.get_label_row_from_db()

    def get_label_row_json(self, cache_db: Optional[prisma.Prisma] = None) -> Dict[str, Any]:
        entry = self.get_label_row_from_db(cache_db=cache_db)
        label_row_json = entry.label_row_json
        if label_row_json is None:
            raise ValueError(f"label_row_json does not exist (label_hash={self._label_hash}, row={entry is not None})")
        return json.loads(label_row_json)

    @property
    def label_row_json(self) -> Dict[str, Any]:
        return self.get_label_row_json()

    @property
    def label_hash(self) -> str:
        return self._label_hash

    def set_label_row_json(self, label_row_json: Dict[str, Any], cache_db: Optional[prisma.Prisma] = None) -> None:
        with PrismaConnection(self._project, cache_db=cache_db) as conn:
            conn.labelrow.update(
                where={"label_hash": self._label_hash}, data={"label_row_json": json.dumps(label_row_json)}
            )
        # Mark out of date.
        self._label_row = None

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
                        **row,  # type: ignore
                        "label_row_json": label_row_json,
                        "local_path": local_path,
                    },
                    include={
                        "data_units": True,
                    },
                )

    def iter_data_unit(
        self,
        data_unit_hash: Optional[str] = None,
        frame: Optional[int] = None,
        cache_db: Optional["prisma.Prisma"] = None,
    ) -> Iterator[DataUnitStructure]:
        with PrismaConnection(self._project, cache_db=cache_db) as conn:
            where: "prisma.types.LabelRowWhereInput" = {"label_hash": {"equals": self._label_hash}}
            du_where: "prisma.types.DataUnitWhereInput" = {}
            if data_unit_hash is not None:
                du_hash = data_unit_hash  # FIXME: , data_unit_hash)
                du_where["data_hash"] = du_hash
            if frame is not None:
                du_where["frame"] = frame
            all_rows = conn.labelrow.find_many(
                where=where,
                include={
                    "data_units": {
                        "where": du_where,
                    }
                },
            )

            # Check to see if any data needs to be fetched from encord api
            cached_signed_urls = self._project.cached_signed_urls
            requre_remote_image_fetch = any(
                du.data_uri is None and du.data_hash not in cached_signed_urls
                for lr in all_rows
                for du in lr.data_units or []
            )

            # Create encord client if it will be needed in the future.
            encord_project_metadata = self._project.load_project_meta()
            encord_project = None
            if encord_project_metadata.get("has_remote", False) and requre_remote_image_fetch:
                encord_project = get_encord_project_cached(
                    encord_project_metadata["ssh_key_path"], encord_project_metadata["project_hash"]
                )

            for label_row in all_rows:
                data_links = {}
                if encord_project is not None:
                    if any(
                        data_unit.data_uri is None and data_unit.data_hash not in cached_signed_urls
                        for data_unit in label_row.data_units or []
                    ):
                        data_links = encord_project.get_label_row(
                            label_row.label_hash,
                            get_signed_url=True,
                        )
                for data_unit in label_row.data_units or []:
                    du_hash = data_unit.data_hash
                    new_du_hash = self._rev_mappings.get(du_hash, du_hash)
                    if data_unit.data_uri is not None:
                        signed_url = data_unit.data_uri
                    elif data_unit.data_hash in cached_signed_urls:
                        signed_url = cached_signed_urls[data_unit.data_hash]
                    else:
                        data_units = data_links.get("data_units", None)
                        if data_units is not None:
                            signed_url = data_units[du_hash]["data_link"]
                            cached_signed_urls[data_unit.data_hash] = signed_url
                        else:
                            raise RuntimeError("Missing data_uri & not encord project")
                    data_type = label_row.data_type
                    label_hash = label_row.label_hash
                    if label_hash is None:
                        raise RuntimeError("Missing label_hash in prisma")
                    yield DataUnitStructure(
                        label_hash,
                        new_du_hash,
                        data_type,
                        signed_url,
                        data_unit.frame,
                        data_unit.data_hash,
                        data_unit.width,
                        data_unit.height,
                        data_unit.fps,
                    )

    def iter_data_unit_with_image(
        self,
        data_unit_hash: Optional[str] = None,
        frame: Optional[int] = None,
        cache_db: Optional["prisma.Prisma"] = None,
    ) -> Iterator[Tuple[DataUnitStructure, Image.Image]]:
        # Temporary directory for all video decodes
        label_row_json = self.get_label_row_json(cache_db=cache_db)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for data_unit_struct in self.iter_data_unit(data_unit_hash=data_unit_hash, frame=frame, cache_db=cache_db):
                if data_unit_struct.data_type in {"img_group", "image"}:
                    image = download_image(
                        data_unit_struct.signed_url,
                        project_dir=self._project.project_dir,
                    )
                    yield data_unit_struct, image
                elif data_unit_struct.data_type == "video":
                    video_dir = tmp_path / data_unit_struct.du_hash
                    video_dir.mkdir(parents=True, exist_ok=True)
                    images_dir = video_dir / "frames"
                    existing_image = next(
                        images_dir.glob(f"{data_unit_struct.du_hash}_{data_unit_struct.frame}.*"), None
                    )
                    if existing_image is not None:
                        yield data_unit_struct, Image.open(existing_image)
                    else:
                        video_file = video_dir / label_row_json["data_title"]
                        download_file(
                            data_unit_struct.signed_url, project_dir=self._project.project_dir, destination=video_file
                        )
                        extract_frames(video_file, images_dir, data_unit_struct.du_hash)
                    downloaded_image = next(images_dir.glob(f"{data_unit_struct.du_hash}_{data_unit_struct.frame}.*"))
                    yield data_unit_struct, Image.open(downloaded_image)
                else:
                    raise RuntimeError("Unsupported data type")

    def iter_data_unit_with_image_or_signed_url(
        self,
        data_unit_hash: Optional[str] = None,
        frame: Optional[int] = None,
        cache_db: Optional["prisma.Prisma"] = None,
    ) -> Iterator[Tuple[DataUnitStructure, Union[str, Image.Image]]]:
        # Temporary directory for all video decodes
        label_row_json = self.get_label_row_json(cache_db=cache_db)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for data_unit_struct in self.iter_data_unit(data_unit_hash=data_unit_hash, frame=frame, cache_db=cache_db):
                if data_unit_struct.data_type in {"img_group", "image"}:
                    yield data_unit_struct, data_unit_struct.signed_url
                elif data_unit_struct.data_type == "video":
                    # Shared image loading logic
                    video_dir = tmp_path / data_unit_struct.du_hash
                    video_dir.mkdir(parents=True, exist_ok=True)
                    images_dir = video_dir / "frames"
                    existing_image = next(
                        images_dir.glob(f"{data_unit_struct.du_hash}_{data_unit_struct.frame}.*"), None
                    )
                    if existing_image is not None:
                        yield data_unit_struct, Image.open(existing_image)
                    else:
                        video_file = video_dir / label_row_json["data_title"]
                        download_file(data_unit_struct.signed_url, project_dir=self._project, destination=video_file)
                        extract_frames(video_file, images_dir, data_unit_struct.du_hash)
                    downloaded_image = next(images_dir.glob(f"{data_unit_struct.du_hash}_{data_unit_struct.frame}.*"))
                    yield data_unit_struct, Image.open(downloaded_image)
                else:
                    raise RuntimeError(f"Unsupported data type: {data_unit_struct.data_type}")


# Needed for efficient visualisation, otherwise each session gets a different cache state
GLOBAL_CACHED_SIGNED_URLS: Dict[str, str] = {}


class ProjectFileStructure(BaseProjectFileStructure):
    def __init__(self, project_dir: Union[str, Path]):
        super().__init__(project_dir)
        self._mappings = json.loads(self.mappings.read_text()) if self.mappings.exists() else {}
        self._cached_project_meta: "Optional[ProjectMeta]" = None
        self.cached_signed_urls: Dict[str, str] = GLOBAL_CACHED_SIGNED_URLS

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
        # FIXME - where is this needed?: label_hash = self._mappings.get(label_hash, label_hash)
        return LabelRowStructure(mappings=self._mappings, label_hash=label_hash, project=self, label_row=None)

    def data_units(
        self,
        where: Optional["prisma.types.DataUnitWhereInput"] = None,
        include_label_row: bool = False,
        cache_db: Optional[prisma.Prisma] = None,
    ) -> List["prisma.models.DataUnit"]:
        from prisma.types import DataUnitInclude

        to_include = DataUnitInclude(label_row=True) if include_label_row else None
        with PrismaConnection(self, cache_db=cache_db) as conn:
            return conn.dataunit.find_many(where=where, include=to_include, take=1)

    def label_rows(self, cache_db: Optional[prisma.Prisma] = None) -> List["prisma.models.LabelRow"]:
        with PrismaConnection(self, cache_db=cache_db) as conn:
            return conn.labelrow.find_many()

    def iter_labels(self, cache_db: Optional[prisma.Prisma] = None) -> Iterator[LabelRowStructure]:
        label_rows = self.label_rows(cache_db=cache_db)
        if len(label_rows) == 0:
            for label_row_legacy in self.data_legacy_folder.iterdir():
                label_hash = label_row_legacy.name
                if label_hash.startswith("."):
                    continue
                yield LabelRowStructure(mappings=self._mappings, label_hash=label_hash, project=self, label_row=None)
        else:
            for label_row in label_rows:
                label_hash_opt = label_row.label_hash
                if label_hash_opt is None:
                    continue
                yield LabelRowStructure(
                    mappings=self._mappings, label_hash=label_hash_opt, project=self, label_row=label_row
                )

    @property
    def mappings(self) -> Path:
        return self.project_dir / "hash_mappings.json"

    def __repr__(self) -> str:
        return f"ProjectFileStructure({self.project_dir})"

    def __hash__(self):
        return hash(self.project_dir)

    def __eq__(self, other):
        return isinstance(other, BaseProjectFileStructure) and other.project_dir == self.project_dir


def is_workflow_project(pfs: ProjectFileStructure):
    label_row_meta: dict[str, dict] = json.loads(pfs.label_row_meta.read_text(encoding="utf-8"))
    return len(label_row_meta) > 0 and next(iter(label_row_meta.values())).get("workflow_graph_node") is not None
