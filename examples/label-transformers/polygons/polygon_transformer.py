import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from encord_active.lib.labels.label_transformer import (
    DataLabel,
    LabelTransformer,
    PolygonLabel,
)


def get_meta_and_labels(label_files: List[Path], extension: str = ".json"):
    meta_file = next((lf for lf in label_files if "meta" in lf.stem), None)

    if meta_file is None:
        raise ValueError("No meta file provided")
    meta = json.loads(meta_file.read_text())["videos"]

    label_files = [lf for lf in label_files if lf.suffix == extension and lf != meta_file]

    return meta, label_files


def label_file_to_image(label_file: Path) -> Path:
    return label_file.parents[2] / "JPEGImages" / label_file.parent.name / f"{label_file.stem}.jpg"


class PolyTransformer(LabelTransformer):
    def from_custom_labels(self, label_files: List[Path], data_files: List[Path]) -> List[DataLabel]:
        meta, label_files = get_meta_and_labels(label_files, extension=".png")

        out = []
        for label_file in label_files:
            classes = meta[label_file.parent.name]["objects"]
            image_file = label_file_to_image(label_file)

            image = np.asarray(Image.open(label_file))

            h, w = image.shape[:2]
            normalization = np.array([[w, h]], dtype=float)

            for instance_id in np.unique(image):
                if instance_id == 0:
                    continue

                instance_mask = (image == instance_id).astype(np.uint8)
                contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    contour = contour.squeeze() / normalization

                    out.append(
                        DataLabel(
                            abs_data_path=image_file,
                            label=PolygonLabel(
                                class_=classes.get(str(instance_id), {}).get("category", "unknown"),
                                polygon=contour,
                            ),
                        )
                    )
                    break

        return out
