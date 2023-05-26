import typing
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Union

if typing.TYPE_CHECKING:
    import prisma


class BaseProjectFileStructure:
    def __init__(self, project_dir: Union[str, Path]):
        if isinstance(project_dir, str):
            project_dir = Path(project_dir)
        self.project_dir: Path = project_dir.expanduser().resolve()

    @abstractmethod
    def cache_clear(self) -> None:
        """
        Clear any cached data and fetch the latest changes.
        """
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
    def prisma_db(self) -> Path:
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

    @abstractmethod
    def data_units(
        self, where: Optional[List["prisma.types.DataUnitWhereInput"]] = None, include_label_row: bool = False
    ) -> List["prisma.models.DataUnit"]:
        pass

    @abstractmethod
    def label_rows(self) -> List["prisma.models.LabelRow"]:
        pass
