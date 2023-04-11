from abc import abstractmethod
from pathlib import Path
from typing import Iterator, Union

from encord_active.lib.db.base import DataUnit


class BaseProjectFileStructure:
    def __init__(self, project_dir: Union[str, Path]):
        if isinstance(project_dir, str):
            project_dir = Path(project_dir)
        self.project_dir: Path = project_dir.expanduser().resolve()

    @property
    @abstractmethod
    def data(self) -> Path:
        pass

    @property
    @abstractmethod
    def metrics(self) -> Path:
        pass

    @property
    @abstractmethod
    def metrics_meta(self) -> Path:
        pass

    @property
    @abstractmethod
    def embeddings(self) -> Path:
        pass

    @property
    @abstractmethod
    def predictions(self) -> Path:
        pass

    @property
    @abstractmethod
    def db(self) -> Path:
        pass

    @property
    @abstractmethod
    def label_row_meta(self) -> Path:
        pass

    @property
    @abstractmethod
    def image_data_unit(self) -> Path:
        pass

    @property
    @abstractmethod
    def ontology(self) -> Path:
        pass

    @property
    @abstractmethod
    def project_meta(self) -> Path:
        pass

    @property
    @abstractmethod
    def label_rows(self) -> Iterator:
        pass

    @property
    @abstractmethod
    def data_units(self) -> Iterator[DataUnit]:
        pass
