import json
import os.path
import pickle

import cv2
import numpy as np
import torch
from tqdm import tqdm
from utils.encord_dataset import EncordMaskRCNNDataset
from utils.model_libs import get_model_instance_segmentation
from utils.provider import get_config, get_transform

from encord_active.lib.db.predictions import Format, Prediction

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
confidence_threshold = 0.5

predictions_to_store = []

params = get_config("config.ini")

with open(params.inference.ontology_filepath, encoding="utf-8") as f:
    encord_ontology: dict = json.load(f)

dataset_validation = EncordMaskRCNNDataset(
    img_folder=params.inference.target_data_folder,
    ann_file=params.inference.target_ann,
    transforms=get_transform(train=False),
)

model = get_model_instance_segmentation(len(dataset_validation.coco.cats) + 1)
model.load_state_dict(torch.load(params.inference.model_checkpoint_path))
model.to(device)

model.eval()
with torch.no_grad():
    for img, _, img_metadata in tqdm(dataset_validation, desc="Generating Encord Predictions"):
        prediction = model([img.to(device)])

        scores_filter = prediction[0]["scores"] > confidence_threshold
        masks = prediction[0]["masks"][scores_filter].detach().cpu().numpy()
        labels = prediction[0]["labels"][scores_filter].detach().cpu().numpy()
        scores = prediction[0]["scores"][scores_filter].detach().cpu().numpy()

        for ma, la, sc in zip(masks, labels, scores):
            contours, hierarchy = cv2.findContours((ma[0] > 0.5).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                contour = contour.reshape(contour.shape[0], 2) / np.array([[ma.shape[2], ma.shape[1]]])
                prediction = Prediction(
                    data_hash=img_metadata[0]["data_hash"],
                    class_id=encord_ontology["objects"][la.item() - 1]["featureNodeHash"],
                    confidence=sc.item(),
                    format=Format.POLYGON,
                    data=contour,
                )
                predictions_to_store.append(prediction)

with open(os.path.join(params.inference.target_data_folder, f"predictions_{params.inference.wandb_id}.pkl"), "wb") as f:
    pickle.dump(predictions_to_store, f)
