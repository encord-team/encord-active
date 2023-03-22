import json
from pathlib import Path
from typing import List


def get_meta_and_labels(label_files: List[Path], extension: str = ".json"):
    meta_file = next((lf for lf in label_files if "meta" in lf.stem), None)

    if meta_file is None:
        raise ValueError("No meta file provided")
    meta = json.loads(meta_file.read_text())["videos"]

    label_files = [lf for lf in label_files if lf.suffix == extension and lf != meta_file]

    return meta, label_files


def label_file_to_image(label_file: Path) -> Path:
    return label_file.parents[2] / "JPEGImages" / label_file.parent.name / f"{label_file.stem}.jpg"
