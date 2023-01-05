import csv
from pathlib import Path
from typing import Optional, Union

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.writer import CSVWriter

SCORE_CSV_FIELDS = ["identifier", "score", "description", "object_class", "annotator", "frame", "url"]


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
