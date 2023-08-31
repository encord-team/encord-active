import csv
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.writer import CSVWriter

SCORE_CSV_FIELDS = ["identifier", "score", "description", "object_class", "annotator", "frame", "url"]

logger.remove()
logger.add(
    sys.stderr,
    format="<e>{time}</e> | <e>{level}</e> | {message} | <e><d>{extra}</d></e>",
    colorize=True,
    level="DEBUG",
)


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
        logger.debug("Metric writer writing score", score=score, filename=self.filename)
        if not isinstance(score, (float, int)):
            raise TypeError("score must be a float or int")
        if isinstance(labels, list) and len(labels) == 0:
            labels = None
        elif isinstance(labels, dict):
            labels = [labels]

        if labels is None:
            label_class = "" if label_class is None else label_class
            annotator = "" if annotator is None else annotator
        else:
            label_class = labels[0]["name"] if label_class is None else label_class
            annotator = labels[0]["lastEditedBy"] if "lastEditedBy" in labels[0] else labels[0]["createdBy"]

        # remember to remove if clause (not its content) when writer's format (obj, score) is enforced on all metrics
        # start hack
        url = ""
        if key is None:
            label_hash = self.iterator.label_hash if label_hash is None else label_hash
            du_hash = self.iterator.du_hash if du_hash is None else du_hash
            frame = self.iterator.frame if frame is None else frame
            url = self.iterator.get_data_url() if url is None else url
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


class CSVVideoAverageWrapper(CSVWriter):
    def __init__(self, metric_writer: CSVMetricWriter):
        filename = metric_writer.filename.parent / f"{metric_writer.filename.stem}_avg{metric_writer.filename.suffix}"
        super(CSVVideoAverageWrapper, self).__init__(filename=filename, iterator=metric_writer.iterator)

        self.writer = csv.DictWriter(self.csv_file, fieldnames=SCORE_CSV_FIELDS)
        self.wrapped_writer = metric_writer

        self.records: dict[tuple[str, str], list[Union[float, int]]] = {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.writeheader()
        logger.debug("records", records=self.records)
        for (lh, dh), vals in self.records.items():
            logger.debug(lh, dh, vals)
            if len(vals) < 1:
                return
            key = self.get_identifier(labels=None, label_hash=lh, du_hash=dh, frame=0)
            row = {
                "identifier": key,
                "score": np.mean(vals).item(),
                "description": "",
                "object_class": "",
                "frame": 0,
                "url": f"",
                "annotator": "",
            }
            self.writer.writerow(row)

        self.csv_file.flush()
        logger.debug(f"Flushing", self.wrapped_writer.filename)
        return super().__exit__(exc_type, exc_val, exc_tb)

    def write(
        self,
        score: Union[float, int],
        key: Optional[str] = None,
    ):
        logger.debug("Video writer", score=score, key=key)
        if key is not None and len(key) > 0:
            label_hash, du_hash, *_ = key.split("_")
            self.records.setdefault((label_hash, du_hash), []).append(score)
            self.wrapped_writer.write(score, key=key)
        return super().write(score)


class CSVAvgScoreWriter(CSVWriter):
    def __init__(self, metric_writer: CSVMetricWriter):
        filename = metric_writer.filename.parent / f"{metric_writer.filename.stem}_avg{metric_writer.filename.suffix}"
        super(CSVAvgScoreWriter, self).__init__(filename=filename, iterator=metric_writer.iterator)

        self.writer = csv.DictWriter(self.csv_file, fieldnames=SCORE_CSV_FIELDS)
        self.writer.writeheader()
        self.metric_writer = metric_writer
        self.records: dict[tuple[str, str], list[Union[float, int]]] = {}

    def record(
        self,
        score: Union[float, int],
    ):
        self.metric_writer.write(score)

        label_hash = self.iterator.label_hash
        du_hash = self.iterator.du_hash
        self.records.setdefault((label_hash, du_hash), []).append(score)

    def write(self):
        for (label_hash, du_hash), values in self.records.items():
            if len(values) < 2:
                return

            key = self.get_identifier(labels=None, label_hash=label_hash, du_hash=du_hash, frame=0)
            row = {
                "identifier": key,
                "score": sum(values) / len(values),
                "description": "",
                "object_class": "",
                "frame": 0,
                "url": f"https://app.encord.com/label_editor/{du_hash}&{self.iterator.project.project_hash}",
                "annotator": "",
            }
            self.writer.writerow(row)
            self.csv_file.flush()
