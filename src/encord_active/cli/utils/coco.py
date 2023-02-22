import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from encord_active.cli.config import app_config
from encord_active.lib.coco.importer import IMAGE_DATA_UNIT_FILENAME, CocoImporter
from encord_active.lib.coco.parsers import parse_results
from encord_active.lib.db.predictions import (
    BoundingBox,
    Format,
    ObjectDetection,
    Prediction,
)
from encord_active.lib.metrics.execute import run_metrics


def import_coco_predictions(
    target: Path,
    predictions_path: Path,
):
    image_data_unit = json.loads((target / IMAGE_DATA_UNIT_FILENAME).read_text(encoding="utf-8"))
    ontology = json.loads((target / "ontology.json").read_text(encoding="utf-8"))
    category_to_hash = {obj["id"]: obj["featureNodeHash"] for obj in ontology["objects"]}
    predictions = []
    results = json.loads(predictions_path.read_text(encoding="utf-8"))

    coco_results = parse_results(results)
    for res in coco_results:
        image = image_data_unit[str(res.image_id)]
        img_h, img_w = image["height"], image["width"]

        if res.segmentation is not None:
            format = Format.POLYGON
            data = res.segmentation / np.array([[img_h, img_w]])
        elif res.bbox:
            format = Format.BOUNDING_BOX
            x, y, w, h = res.bbox
            data = BoundingBox(x=x / img_w, y=y / img_h, w=w / img_w, h=h / img_h)
        else:
            raise Exception("Unsupported result format")

        predictions.append(
            Prediction(
                data_hash=image["data_hash"],
                confidence=res.score,
                object=ObjectDetection(
                    feature_hash=category_to_hash[str(res.category_id)],
                    format=format,
                    data=data,
                ),
            )
        )

    return predictions


def import_coco_project(images_dir: Path, annotations_file: Path, target: Path, use_symlinks: bool = False) -> Path:
    """
    Importer for COCO datasets.
    Args:
        images_dir (Path): path where images are stored
        annotations_file (Path): the COCO JSON annotation file
        target (Path): where to store the data
        use_symlinks (bool): if False, the importer will copy images.
            Otherwise, symlinks will be used to save disk space.
    """
    coco_importer = CocoImporter(
        images_dir_path=images_dir,
        annotations_file_path=annotations_file,
        destination_dir=target,
        use_symlinks=use_symlinks,
    )

    with TemporaryDirectory() as temp_dir:
        temp_folder = Path(temp_dir)

        dataset = coco_importer.create_dataset(temp_folder)
        ontology = coco_importer.create_ontology()
        coco_importer.create_project(
            dataset=dataset,
            ontology=ontology,
            ssh_key_path=app_config.get_ssh_key(),
        )

    run_metrics(data_dir=coco_importer.project_dir, use_cache_only=True)

    return coco_importer.project_dir
