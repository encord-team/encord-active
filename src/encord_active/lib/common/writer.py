import math
import typing
from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Optional, Union
from typing_extensions import Self

import numpy as np

from encord_active.lib.project import ProjectFileStructure

if typing.TYPE_CHECKING:
    import prisma

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.db.connection import PrismaConnection


class MetricObserver(ABC):
    @abstractmethod
    def on_value_insert(self, value: Union[float, int, list]):
        pass

    @abstractmethod
    def on_metric_close(self):
        pass


class StatisticsObserver(MetricObserver):
    def __init__(self):
        self.min_value = math.inf
        self.max_value = -math.inf
        self.num_rows = 0
        self.mean_value = 0

    def on_value_insert(self, value: Union[float, int, list]):
        if isinstance(value, list):
            value = float(np.linalg.norm(np.array(value)))
        elif not isinstance(value, (int, float)):
            raise TypeError(f"Expected float, int, or list, got {type(value)}")

        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.mean_value = (self.mean_value * self.num_rows + value) / (self.num_rows + 1)
        self.num_rows += 1

    def on_metric_close(self):
        pass


class Writer(ABC):
    def __init__(self):
        self._observers: List[MetricObserver] = []

    def attach(self, observer: MetricObserver):
        self._observers.append(observer)

    def remove_listener(self, observer: MetricObserver):
        self._observers.remove(observer)

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        for observer in self._observers:
            observer.on_metric_close()

    def write(self, value):
        for observer in self._observers:
            observer.on_value_insert(value)


class DBWriter(Writer):
    def __init__(self, project_file_structure: "ProjectFileStructure", iterator: Iterator) -> None:
        super().__init__()
        self.project_file_structure = project_file_structure
        self.iterator = iterator
        self._prisma_db_conn: Optional[PrismaConnection] = None
        self._conn: Optional["prisma.Prisma"]

    def __enter__(self) -> Self:
        self._prisma_db_conn = PrismaConnection(self.project_file_structure)
        self._conn = self._prisma_db_conn.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._conn = None
        self._prisma_db_conn.__exit__(exc_type, exc_val, exc_tb)

    def get_identifier(
        self,
        labels: Union[list[dict], dict, None] = None,
        label_hash: Optional[str] = None,
        du_hash: Optional[str] = None,
        frame: Optional[int] = None,
    ):
        label_hash = self.iterator.label_hash if label_hash is None else label_hash
        du_hash = self.iterator.du_hash if du_hash is None else du_hash
        frame = self.iterator.frame if frame is None else frame

        identifier = f"{label_hash}_{du_hash}_{frame:05d}"

        if labels is not None:
            if isinstance(labels, dict):
                labels = [labels]
            hashes = [lbl["objectHash"] if "objectHash" in lbl else lbl["featureHash"] for lbl in labels]
            return "_".join(chain([identifier], hashes))
        return identifier
