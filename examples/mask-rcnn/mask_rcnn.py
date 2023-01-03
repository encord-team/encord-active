import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from config_utils import flatten_dict
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from encord_active.app.db.predictions import Format, Prediction


def threshold_masks(outputs: List[Dict[str, torch.Tensor]], threshold: float = 0.5):
    thresh_outs = []
    for out in outputs:
        out["masks"] = (out["masks"] > threshold).squeeze()
        if len(out["masks"].shape) == 2:
            out["masks"] = out["masks"].unsqueeze(0)
        thresh_outs.append(out)
    outputs = thresh_outs
    return outputs


def load_model(num_classes: int) -> torch.nn.Module:
    """
    Load a MaskRCNN model with pretrained weights and replace the head with a new one
    Args:
        num_classes: Number of classes to predict

    Returns:
        MaskRCNN model
    """
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    # get number of input features for the classifier and replace the pre-trained head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # get input features for the mask and replace the mask predictor with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced=256, num_classes=num_classes)
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.roi_heads.box_predictor.parameters():
    #     param.requires_grad = True
    # for param in model.roi_heads.mask_predictor.parameters():
    #     param.requires_grad = True
    return model


def get_ea_predictions(
    outputs: List[Dict], encord_ontology: dict, confidence_threshold: float = 0.5
) -> List[Prediction]:
    """
    Convert the outputs of the model to a list of Encord Active predictions
    Args:
        outputs: List of outputs from the Mask RCNN model
        encord_ontology: ontology of the Encord project
        confidence_threshold: Threshold for the confidence of the predictions to be included

    Returns:
        List of Encord Active predictions
    """
    predictions_to_store = []
    for preds, metadata in outputs:
        for i, (im_out, im_meta) in enumerate(zip(preds, metadata)):
            scores_filter = im_out["scores"] > confidence_threshold
            masks = im_out["masks"][scores_filter].detach().cpu().numpy()
            labels = im_out["labels"][scores_filter].detach().cpu().numpy()
            scores = im_out["scores"][scores_filter].detach().cpu().numpy()

            for ma, la, sc in zip(masks, labels, scores):
                contours, hierarchy = cv2.findContours(ma.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    contour = contour.reshape(contour.shape[0], 2) / np.array([[ma.shape[0], ma.shape[1]]])
                    prediction = Prediction(
                        data_hash=im_meta["data_hash"],
                        class_id=encord_ontology["objects"][la.item() - 1]["featureNodeHash"],
                        confidence=sc.item(),
                        format=Format.POLYGON,
                        data=contour,
                    )
                    predictions_to_store.append(prediction)
    return predictions_to_store


class MaskRCNN(pl.LightningModule):
    def __init__(
        self,
        params: SimpleNamespace,
        num_classes: int,
        confidence_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(flatten_dict(vars(params)))

        self.val_data_root = Path(params.data.val_data_root).expanduser()
        self.confidence_threshold = confidence_threshold

        self.train_map = MeanAveragePrecision(iou_type="segm")
        self.val_map = MeanAveragePrecision(iou_type="segm")
        self.test_map = MeanAveragePrecision(iou_type="segm")

        self.model = load_model(num_classes)
        self.params = params

        with open(params.general.ontology_file_path, encoding="utf-8") as f:
            self.encord_ontology: dict = json.load(f)

        self.best_val_map = 0

    def training_step(self, batch, batch_idx):
        images, targets, metadata = batch
        losses = self.model(images, targets)
        losses["total"] = sum(loss for loss in losses.values())
        self.log_dict({f"train/{k}": v.item() for k, v in losses.items()})

        return losses["total"]

    def validation_step(self, batch, batch_idx):
        images, targets, metadata = batch
        preds = self.model(images)

        preds = threshold_masks(preds, threshold=self.confidence_threshold)
        targets = threshold_masks(targets, threshold=self.confidence_threshold)

        if self.current_epoch % 3 == 0:
            self.val_map.update(preds=preds, target=targets)

        return preds, metadata

    def validation_epoch_end(self, outputs):
        if self.current_epoch % 5 == 0:
            map_dict = self.val_map.compute()
            self.val_map.reset()
            map_dict = {f"val/{k}": v.item() for k, v in map_dict.items()}
            self.log_dict(map_dict)

        # # Uncomment later
        # if map_dict["val/map"] > self.best_val_map:
        #     self.best_val_map = map_dict["val/map"]
        #     predictions_to_store = get_ea_predictions(
        #         outputs,
        #         self.encord_ontology,
        #         confidence_threshold=self.confidence_threshold,
        #     )
        #
        #     pred_path = self.val_data_root.joinpath(f"val_predictions_epoch{self.current_epoch}.pkl")
        #
        #     with open(pred_path.as_posix(), "wb") as f:
        #         pickle.dump(predictions_to_store, f)
        #
        #     print(f"Stored predictions file at {pred_path}")

    # def test_step(self, batch, batch_idx, dataloader_idx=None):
    #     images, targets, metadata = batch
    #     preds = self.model(images)
    #
    #     preds = threshold_masks(preds)
    #     targets = threshold_masks(targets)
    #
    #     self.test_map.update(preds=preds, target=targets)
    #     return dataloader_idx
    #
    # def test_epoch_end(self, outputs):
    #     map_dict = self.test_map.compute()
    #     map_dict = {f"test_{outputs[0]}/{k}": v.item() for k, v in map_dict.items()}
    #     self.log_dict(map_dict)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.params.train.lr)
        return optimizer
