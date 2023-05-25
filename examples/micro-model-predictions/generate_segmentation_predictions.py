import configparser
import json
import pickle
from pathlib import Path

import numpy as np
from encord import EncordUserClient, Project
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
ontology_featureHashes = [obj["featureNodeHash"] for obj in ontology.get("objects")]

predictions_to_store = []
file_paths = []
data_units = []

pbar = tqdm(total=iterator.length, desc="Running inference", leave=False)
for counter, (data_unit, _) in enumerate(iterator.iterate()):

    file_paths.append(
        next(
            ea_project_fs.label_row_structure(iterator.label_hash).iter_data_unit(data_unit["data_hash"])
        ).path.as_posix()
    )
    data_units.append(data_unit)

    if (counter + 1) % int(params["BATCH_SIZE"]) == 0 or counter + 1 == iterator.length:
        inference_results = project.model_inference(
            params["ENCORD_MODEL_ITERATION_HASH"],
            file_paths=file_paths,
            conf_thresh=float(params["CONFIDENCE_THRESHOLD"]),
            iou_thresh=float(params["IOU_THRESHOLD"]),
        )

        for inference_result, du in zip(inference_results, data_units):
            for obj in inference_result["predictions"]["0"]["objects"]:

                if obj["featureHash"] not in ontology_featureHashes:
                    print(
                        f"'{obj['name']}' with featureHash '{obj['featureHash']}' is not available in the ontology of"
                        f" the Encord Active project."
                    )
                    continue

                polygon_points = [[key, [value["x"], value["y"]]] for key, value in obj["polygon"].items()]
                polygon_points_sorted = sorted(polygon_points, key=lambda x: int(x[0]))
                polygon = np.array([item[1] for item in polygon_points_sorted])

                prediction = Prediction(
                    data_hash=du["data_hash"],
                    confidence=obj["confidence"],
                    object=ObjectDetection(
                        format=Format.POLYGON,
                        data=polygon,
                        feature_hash=obj["featureHash"],
                    ),
                )
                predictions_to_store.append(prediction)

        file_paths = []
        data_units = []

    pbar.update(1)

with open((ea_project_fs.project_dir / f"predictions_{params['ENCORD_MODEL_ITERATION_HASH'][:8]}.pkl"), "wb") as f:
    pickle.dump(predictions_to_store, f)
