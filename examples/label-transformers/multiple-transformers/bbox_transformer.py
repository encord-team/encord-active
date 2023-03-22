import json
from pathlib import Path
from typing import List

from encord_active.lib.labels.label_transformer import (
    BoundingBox,
    BoundingBoxLabel,
    DataLabel,
    LabelTransformer,
)
from examples.utils import get_meta_and_labels, label_file_to_image


class BBoxTransformer(LabelTransformer):
    def from_custom_labels(self, label_files: List[Path], data_files: List[Path]) -> List[DataLabel]:
        meta, label_files = get_meta_and_labels(label_files, extension=".json")

        out = []
        for label_file in label_files:
            classes = meta[label_file.parent.name]["objects"]

            labels = json.loads(label_file.read_text())
            image_file = label_file_to_image(label_file)

            for instance_id, bbox in labels.items():
                out.append(
                    DataLabel(
                        abs_data_path=image_file,
                        label=BoundingBoxLabel(
                            class_=classes.get(instance_id, {}).get("category", "unknown"),
                            bounding_box=BoundingBox(**bbox),
                        ),
                    )
                )
        return out
