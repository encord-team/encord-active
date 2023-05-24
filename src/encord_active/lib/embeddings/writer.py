import json
from typing import Optional, Union

from prisma import Base64

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.writer import DBWriter

EMBEDDING_CSV_FIELDS = ["identifier", "embedding", "description", "object_class", "frame", "url"]


class DBEmbeddingWriter(DBWriter):
    def __init__(self, project_file_structure: "ProjectFileStructure", iterator: Iterator, prefix: str):
        super().__init__(project_file_structure, iterator)
        self.prefix = prefix

    def write(
        self,
        value: Union[float, list],
        labels: Union[list[dict], dict, None] = None,
        description: str = "",
        label_class: Optional[str] = None,
        label_hash: Optional[str] = None,
        du_hash: Optional[str] = None,
        frame: int = 0,
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
        url = "null"  # self.iterator.get_data_url() if url is None else url

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
        identifier = self.get_identifier(labels, label_hash, du_hash, frame)
        values = [value] if isinstance(value, float) else value
        values = [float(value) for value in values]
        embedding_bytes = Base64.encode(json.dumps(values).encode("utf-8"))
        if self._conn is not None:
            self._conn.embeddingrow.upsert(
                where={
                    "metric_prefix_identifier": {
                        "metric_prefix": self.prefix,
                        "identifier": identifier,
                    },
                },
                data={
                    "create": {
                        "metric_prefix": self.prefix,
                        "identifier": identifier,
                        "frame": frame,
                        "embedding": embedding_bytes,
                        "description": description,
                        "object_class": label_class,
                        "url": url,
                    },
                    "update": {
                        "embedding": embedding_bytes,
                        "description": description,
                        "object_class": label_class,
                        "url": url,
                    }
                }
            )
        else:
            raise RuntimeError("Prisma connection is missing")

