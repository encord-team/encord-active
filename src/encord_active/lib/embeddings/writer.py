import csv
from pathlib import Path
from typing import Optional, Union

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.writer import CSVWriter

EMBEDDING_CSV_FIELDS = ["identifier", "embedding", "description", "object_class", "frame", "url"]


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
