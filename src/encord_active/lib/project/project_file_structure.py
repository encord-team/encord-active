from pathlib import Path
from typing import NamedTuple


class LabelRowStructure(NamedTuple):
    path: Path
    images_dir: Path
    label_row_file: Path


class ProjectFileStructure:
    def __init__(self, project_dir: Path):
        self.project_dir: Path = project_dir.expanduser().resolve()

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
    def ontology(self) -> Path:
        return self.project_dir / "ontology.json"

    @property
    def project_meta(self) -> Path:
        return self.project_dir / "project_meta.yaml"

    def label_row_structure(self, label_hash: str) -> LabelRowStructure:
        path = self.data / label_hash
        return LabelRowStructure(path=path, images_dir=path / "images", label_row_file=path / "label_row.json")
