import json
from pathlib import Path

import numpy as np

from encord_active.app.db.predictions import BoundingBox, Format, Prediction
from encord_active.lib.coco.importer import IMAGE_DATA_UNIT_FILENAME, CocoImporter
from encord_active.lib.coco.parsers import parse_results
from encord_active.lib.metrics.run_all import run_metrics


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
                class_id=category_to_hash[str(res.category_id)],
                confidence=res.score,
                format=format,
                data=data,
            )
        )

    return predictions


def import_coco_project(images_dir: Path, annotations_file: Path, target: Path, ssh_key_path: Path):
    coco_importer = CocoImporter(
        images_dir_path=images_dir,
        annotations_file_path=annotations_file,
        ssh_key_path=ssh_key_path,
        destination_dir=target,
    )

    dataset = coco_importer.create_dataset()
    ontology = coco_importer.create_ontology()
    coco_importer.create_project(
        dataset=dataset,
        ontology=ontology,
    )

    run_metrics(data_dir=coco_importer.project_dir)
