import csv
import math
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from encord_active.lib.common.iterator import Iterator

SCORE_CSV_FIELDS = ["identifier", "score", "description", "object_class", "annotator", "frame", "url"]
EMBEDDING_CSV_FIELDS = ["identifier", "embedding", "description", "object_class", "frame", "url"]


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


class CSVMetricWriter(CSVWriter):
    def __init__(self, data_path: Path, iterator: Iterator, prefix: str):
        filename = (data_path / "metrics" / f"{prefix}.csv").expanduser()
        super(CSVMetricWriter, self).__init__(filename=filename, iterator=iterator)

        self.writer = csv.DictWriter(self.csv_file, fieldnames=SCORE_CSV_FIELDS)
        self.writer.writeheader()

    def write(
        self,
        score: Union[float, int],
        labels: Union[list[dict], dict, None] = None,
        description: str = "",
        label_class: Optional[str] = None,
        label_hash: Optional[str] = None,
        du_hash: Optional[str] = None,
        frame: Optional[int] = None,
        url: Optional[str] = None,
        annotator: Optional[str] = None,
        key: Optional[str] = None,  # TODO obsolete parameter, remove from metrics first
    ):
        if not isinstance(score, (float, int)):
            raise TypeError("score must be a float or int")
        if isinstance(labels, list) and len(labels) == 0:
            labels = None
        elif isinstance(labels, dict):
            labels = [labels]

        label_hash = self.iterator.label_hash if label_hash is None else label_hash
        du_hash = self.iterator.du_hash if du_hash is None else du_hash
        frame = self.iterator.frame if frame is None else frame
        url = self.iterator.get_data_url() if url is None else url

        if labels is None:
            label_class = "" if label_class is None else label_class
            annotator = "" if annotator is None else annotator
        else:
            label_class = labels[0]["name"] if label_class is None else label_class
            annotator = labels[0]["lastEditedBy"] if "lastEditedBy" in labels[0] else labels[0]["createdBy"]

        # remember to remove if clause (not its content) when writer's format (obj, score) is enforced on all metrics
        # start hack
        if key is None:
            key = self.get_identifier(labels, label_hash, du_hash, frame)
        # end hack

        row = {
            "identifier": key,
            "score": score,
            "description": description,
            "object_class": label_class,
            "frame": frame,
            "url": url,
            "annotator": annotator,
        }

        self.writer.writerow(row)
        self.csv_file.flush()

        super().write(score)


class CSVEmbeddingWriter(CSVWriter):
    def __init__(self, data_path: Path, iterator: Iterator, prefix: str):
        filename = (data_path / "embeddings" / f"{prefix}.csv").expanduser()
        super(CSVEmbeddingWriter, self).__init__(filename=filename, iterator=iterator)

        self.writer = csv.DictWriter(self.csv_file, fieldnames=EMBEDDING_CSV_FIELDS)
        self.writer.writeheader()

    def write(
        self,
        value: Union[float, list],
        labels: Union[list[dict], dict, None] = None,
        description: str = "",
        label_class: Optional[str] = None,
        label_hash: Optional[str] = None,
        du_hash: Optional[str] = None,
        frame: Optional[int] = None,
        url: Optional[str] = None,
    ):
        if not isinstance(value, (float, list)):
            raise TypeError("value must be a float or list")

        if isinstance(labels, list) and len(labels) == 0:
            labels = None
        elif isinstance(labels, dict):
            labels = [labels]

        label_hash = self.iterator.label_hash if label_hash is None else label_hash
        du_hash = self.iterator.du_hash if du_hash is None else du_hash
        frame = self.iterator.frame if frame is None else frame
        url = self.iterator.get_data_url() if url is None else url

        if labels is None:
            label_class = "" if label_class is None else label_class
        else:
            label_class = labels[0]["name"] if label_class is None else label_class

        row = {
            "identifier": self.get_identifier(labels, label_hash, du_hash, frame),
            "embedding": value,
            "description": description,
            "object_class": label_class,
            "frame": frame,
            "url": url,
        }

        self.writer.writerow(row)
        self.csv_file.flush()

        super().write(value)
