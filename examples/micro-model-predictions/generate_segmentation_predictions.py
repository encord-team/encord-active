import configparser
import json
import pickle
from pathlib import Path

import numpy as np
from encord import EncordUserClient, Project
from encord.constants.model import Device
from tqdm import tqdm

from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.db.predictions import Format, ObjectDetection, Prediction
from encord_active.lib.project.project_file_structure import ProjectFileStructure

config = configparser.ConfigParser()
config.read("config.ini")
params = config["PARAMETERS"]

user_client = EncordUserClient.create_with_ssh_private_key(Path(params["ENCORD_SSH_KEY_PATH"]).expanduser().read_text())
project: Project = user_client.get_project(params["ENCORD_PROJECT_HASH"])

ea_project_fs: ProjectFileStructure = ProjectFileStructure(params["ENCORD_ACTIVE_PROJECT_PATH"])
iterator = DatasetIterator(ea_project_fs.project_dir)

ontology = json.loads(ea_project_fs.ontology.read_text(encoding="utf-8"))
ontology_names = [obj["name"] for obj in ontology.get("objects")]
ontology_name_to_featurehash = {obj["name"]: obj["featureNodeHash"] for obj in ontology.get("objects")}

predictions_to_store = []
for data_unit, _ in tqdm(iterator.iterate()):

    img_path = next(ea_project_fs.label_row_structure(iterator.label_hash).iter_data_unit(data_unit["data_hash"]))

    inference_result = project.model_inference(
        params["ENCORD_MODEL_ITERATION_HASH"],
        file_paths=[img_path.path.as_posix()],
        conf_thresh=float(params["CONFIDENCE_THRESHOLD"]),
        iou_thresh=float(params["IOU_THRESHOLD"]),
    )

    for object in inference_result[0]["predictions"]["0"]["objects"]:

        polygon_points = [[key, [value["x"], value["y"]]] for key, value in object["polygon"].items()]
        polygon_points_sorted = sorted(polygon_points, key=lambda x: int(x[0]))
        polygon = np.array([item[1] for item in polygon_points_sorted])

        prediction = Prediction(
            data_hash=data_unit["data_hash"],
            confidence=object["confidence"],
            object=ObjectDetection(
                format=Format.POLYGON,
                data=polygon,
                feature_hash=object["featureHash"],
            ),
        )
        predictions_to_store.append(prediction)

with open((ea_project_fs.project_dir / "predictions_sam.pkl"), "wb") as f:
    pickle.dump(predictions_to_store, f)
