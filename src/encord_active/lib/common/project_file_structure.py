from pathlib import Path


class ProjectFileStructure:
    def __init__(self, project_dir: Path):
        self.project_dir: Path = project_dir

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

    def get_label_row_file_path(self, label_hash: str) -> Path:
        return self.data / label_hash / "label_row.json"
