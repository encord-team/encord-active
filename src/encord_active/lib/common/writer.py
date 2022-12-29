import math
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from encord_active.lib.common.iterator import Iterator


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
    _observers: List[MetricObserver] = []

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


class CSVWriter(Writer):
    def __init__(self, filename: Path, iterator: Iterator):
        super(CSVWriter, self).__init__()

        self.iterator = iterator

        self.filename = filename
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.filename.open("w", newline="", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.csv_file.close()  # Notify observers
        super(CSVWriter, self).__exit__(exc_type, exc_val, exc_tb)

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
