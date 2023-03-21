import json
from pathlib import Path
from typing import Iterator, NamedTuple, Optional


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

    def iter_data_unit(self, data_unit_hash: Optional[str] = None, frame: Optional[int] = None) -> Iterator[DataUnitStructure]:
        if data_unit_hash:
            glob_string = self._mappings.get(data_unit_hash, data_unit_hash)
        else:
            glob_string = "*"
        if frame:
            glob_string += f"_{frame}"
        glob_string += ".*"
        for du_path in self.images_dir.glob(glob_string):
            du_hash = du_path.name.split('.')[0].split('_')[0]
            yield DataUnitStructure(self._rev_mappings.get(du_hash, du_hash), du_path)

    def is_present(self):
        return self.path.is_dir()


class ProjectFileStructure:
    def __init__(self, project_dir: Path):
        self.project_dir: Path = project_dir.expanduser().resolve()
        self._mappings = (
            json.loads((project_dir / "hash_mappings.json").read_text())
            if (project_dir / "hash_mappings.json").exists()
            else {}
        )

    @property
    def data(self) -> Path:
        return self.project_dir / "data"

    @property
    def metrics(self) -> Path:
        return self.project_dir / "metrics"

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

    def iter_labels(self) -> Iterator[LabelRowStructure]:
        for label_hash in self.data.iterdir():
            path = self.data / label_hash
            yield LabelRowStructure(path=path, mappings=self._mappings)

    @property
    def mappings(self) -> Path:
        return self.project_dir / "hash_mappings.json"
