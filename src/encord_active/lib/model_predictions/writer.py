import json
import logging
from base64 import b64encode
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, TypedDict
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import torch
from encord.objects.common import NestableOption, RadioAttribute
from encord.objects.ontology_structure import Classification, OntologyStructure
from torchvision.ops import box_iou
from tqdm.auto import tqdm

from encord_active.lib.common.utils import RLEData, binary_mask_to_rle, rle_iou
from encord_active.lib.db.predictions import (
    BoundingBox,
    FrameClassification,
    Prediction,
)
from encord_active.lib.labels.classification import (
    ClassificationAnswer,
    LabelClassification,
)
from encord_active.lib.labels.object import BoxShapes, ObjectShape
from encord_active.lib.project import Project

logger = logging.getLogger(__name__)
BBOX_KEYS = {"x", "y", "w", "h"}
BASE_URL = "https://app.encord.com/label_editor/"


class PredictionType(str, Enum):
    FRAME = "frame"
    BBOX = "bounding_box"
    POLYGON = "polygon"


ImageIdentifier = int
# This is a hack to not break backward compatibility
def get_image_identifier(data_hash: str, frame: int) -> ImageIdentifier:
    return abs(hash(f"{data_hash}_{frame}")) % 10_000_000


ClassID = int
PredictionIndex = int
LabelIndex = int


@dataclass
class LabelEntry:
    identifier: str
    url: str
    img_id: ImageIdentifier
    class_id: ClassID  # TODO remove this stupid legacy thing.
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    rle: Optional[RLEData] = None
    theta: Optional[float] = None

    @property
    def bbox_list(self):
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def has_object(self):
        return None not in [self.x1, self.y1, self.x2, self.y2] or self.rle


@dataclass
class PredictionEntry(LabelEntry):
    confidence: float = 0.0


class LabelMatchList(TypedDict):
    lidx: LabelIndex
    pidxs: List[PredictionIndex]


class LabelEntryWithIndex(NamedTuple):
    lidx: LabelIndex
    entry: LabelEntry


class PredictionEntryWithIndex(NamedTuple):
    pidx: PredictionIndex
    entry: PredictionEntry


ImgClsPredictions = List[PredictionEntryWithIndex]
ImgClsLabels = List[LabelEntryWithIndex]
GroundTruthStructure = Dict[Tuple[ImageIdentifier, ClassID], ImgClsLabels]
GroundTruthsMatchedStructure = Dict[ClassID, Dict[ImageIdentifier, List[LabelMatchList]]]


def polyobj_to_nparray(o: dict, width: int, height: int) -> np.ndarray:
    return np.array(
        [(o["polygon"][str(i)]["x"] * width, o["polygon"][str(i)]["y"] * height) for i in range(len(o["polygon"]))]
    )


def points_to_mask(points: np.ndarray, width: int, height: int):
    mask = np.zeros((height, width), dtype=np.uint8)
    if np.issubdtype(points.dtype, np.floating) and points.max() <= 1.0:
        points *= np.array([[width, height]])

    mask = cv2.fillPoly(mask, [(points).astype(int)], 1)  # type: ignore
    return mask


def get_img_ious(detections: ImgClsPredictions, ground_truth_img: ImgClsLabels):
    # Note: we assume that all labels and predictions are of same type, i.e., either segmentations or bboxes.
    if ground_truth_img[0].entry.rle is not None:  # Segmentation
        det_coco_format = [i.entry.rle for i in detections if i.entry.rle]
        gt_coco_format = [i.entry.rle for i in ground_truth_img if i.entry.rle]

        return torch.from_numpy(rle_iou(det_coco_format, gt_coco_format))
    else:
        _pred_boxes = torch.tensor([d.entry.bbox_list for d in detections])  # type: ignore
        _label_boxes = torch.tensor([l.entry.bbox_list for l in ground_truth_img])
        return box_iou(_pred_boxes, _label_boxes)


def precompute_MAP_features(
    pred_boxes: List[PredictionEntry],
    true_boxes: List[LabelEntry],
) -> Tuple[torch.Tensor, Dict[ClassID, Dict[ImageIdentifier, List[LabelMatchList]]]]:
    """
    Calculates mean average precision for given iou threshold and rec_thresholds.
    Parameters:
        pred_boxes: list of lists containing all bboxes.
        true_boxes: Similar as pred_boxes except all the correct ones.

    Returns:
        ious: used as input for the next call to the function.
        ground_truths_matched: used as input for the next call to the function.
    """

    pred_boxes.sort(key=lambda x: x.confidence, reverse=True)

    ground_truths: GroundTruthStructure = {}
    for lidx, true_box in enumerate(true_boxes):
        ground_truths.setdefault((true_box.img_id, true_box.class_id), []).append(LabelEntryWithIndex(lidx, true_box))

    ground_truths_matched: GroundTruthsMatchedStructure = {}
    for (img_id, class_idx), img_cls_labels in ground_truths.items():
        ground_truths_matched.setdefault(class_idx, {})[img_id] = [
            LabelMatchList(lidx=lidx, pidxs=[]) for lidx, _ in img_cls_labels
        ]

    predictions: Dict[Tuple[ImageIdentifier, ClassID], ImgClsPredictions] = {}
    for pidx, pred_box in enumerate(pred_boxes):
        predictions.setdefault((pred_box.img_id, pred_box.class_id), []).append(
            PredictionEntryWithIndex(pidx, pred_box)
        )

    ious: torch.Tensor = torch.zeros(len(pred_boxes), dtype=float)  # type: ignore
    for (img_id, pred_cls), prediction_entries_with_index in tqdm(
        predictions.items(), desc="Matching predictions to labels", leave=True
    ):
        ground_truth_img = ground_truths.get((img_id, pred_cls))
        label_matches = ground_truths_matched.get(pred_cls, {}).get(img_id, [])

        if not ground_truth_img:
            continue

        detections: ImgClsPredictions = []
        classifications: ImgClsPredictions = []
        for pred in prediction_entries_with_index:
            target = detections if pred.entry.has_object else classifications
            target.append(pred)

        for pidx, entry in classifications:
            matches = ground_truths_matched.get(entry.class_id, {}).get(entry.img_id)
            for match in matches or []:
                match["pidxs"].append(pidx)
                # NOTE: we deeply rely on IOUs in the model prediction pages
                ious[pidx] = 1

        if detections:
            img_ious = get_img_ious(detections, ground_truth_img)  # type: ignore
            best_gt_idxs = torch.argmax(img_ious, dim=1, keepdim=False)
            best_ious = torch.amax(img_ious, dim=1, keepdim=False)

            for i, (best_gt_idx, best_iou) in enumerate(zip(best_gt_idxs, best_ious)):
                if best_iou > 0:
                    pidx = detections[i].pidx
                    label_matches[best_gt_idx]["pidxs"].append(pidx)
                    ious[pidx] = best_iou

    return ious, ground_truths_matched


PREDICTIONS_FILENAME = "predictions.csv"
LABELS_FILE = "labels.csv"
GROUND_TRUTHS_MATCHED_FILE = "ground_truths_matched.json"
CLASS_INDEX_FILE = "class_idx.json"


class ClassificationAttributeOption(NamedTuple):
    classification: Classification
    attribute: RadioAttribute
    option: NestableOption


def iterate_classification_attribute_options(ontology: OntologyStructure):
    for classification in ontology.classifications:
        for attribute in classification.attributes:
            if isinstance(attribute, RadioAttribute):
                for option in attribute.options:
                    yield FrameClassification(
                        feature_hash=classification.feature_node_hash,
                        attribute_hash=attribute.feature_node_hash,
                        option_hash=option.feature_node_hash,
                    ), ClassificationAttributeOption(classification, attribute, option)


class PredictionWriter:
    def __init__(
        self,
        project: Project,
        custom_object_map: Optional[Dict[str, int]] = None,
    ):
        logger.info("Fetching project label rows to be able to match predictions.")
        self.project = project.load()
        self.storage_dir = project.file_structure.predictions

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.predictions: List[PredictionEntry] = []
        self.object_lookup = {o.feature_node_hash: o for o in self.project.ontology.objects}

        self.classification_lookup = {
            hashes: option for hashes, (_, _, option) in iterate_classification_attribute_options(self.project.ontology)
        }

        self.label_row_meta = self.project.label_row_metas
        self.uuids: Set[str] = set()

        self.lr_lookup: Dict[str, str] = {m.data_hash: m.label_hash for m in self.project.label_row_metas.values()}
        for lr in self.project.label_rows.values():
            self.lr_lookup.update({du_hash: lr["label_hash"] for du_hash in lr["data_units"]})

        self.__prepare_class_id_lookups(custom_object_map)
        self.__prepare_label_list()

    def __prepare_class_id_lookups(self, custom_map: Optional[Dict[str, ClassID]]):
        """
        TODO: Clean this up. One option is to just have one mapping for everything into a common space.
        So for example a json file describing pairs of "from" -> "to" of class ids.
        This way, anything can be parsed in as values:
            {
                "0": "0",
                "featureNodeHash": "0",
                etc
            }
        """
        logger.debug("Preparing class id lookup")
        self.custom_map = False
        if custom_map:
            feature_hashes = {o.feature_node_hash for o in self.project.ontology.objects}
            custom_hashes = set(custom_map.keys())
            if not all([c in feature_hashes for c in custom_hashes]):
                raise ValueError("custom map keys should correspond to `featureNodeHashes` from the project ontology.")

            self.object_class_id_lookup = custom_map
            self.custom_map = True
        else:
            self.object_class_id_lookup = {}
            for obj in self.project.ontology.objects:
                self.object_class_id_lookup[obj.feature_node_hash] = len(self.object_class_id_lookup)

        self.classification_class_id_lookup = {
            key: index for index, (key, _) in enumerate(iterate_classification_attribute_options(self.project.ontology))
        }

    def __prepare_label_list(self):
        logger.debug("Preparing label list")
        self.object_labels: List[LabelEntry] = []
        self.classification_labels: List[LabelEntry] = []

        def append_classification_label(du_hash: str, frame: int, classification_dict: dict, answers_dict: dict):
            label_hash = self.lr_lookup[du_hash]
            classification = LabelClassification(**classification_dict)

            classification_answers = answers_dict.get(classification.classificationHash, {}).get("classifications", [])
            if not classification_answers:
                return None
            elif len(classification_answers) > 1:
                logger.error(
                    f'Found multiple classifications for label row "{label_hash}" and classification hash "{classification.classificationHash}'
                )

            classification_answer = ClassificationAnswer.parse_obj(classification_answers[0])
            class_id = self.get_classification_class_id(classification, classification_answer)
            if class_id is None:  # Ignore unwanted classes (defined by what is in `self.object_class_id_lookup`)
                return

            label_entry = LabelEntry(
                identifier=f"{label_hash}_{du_hash}_{frame:05d}_{classification.classificationHash}",
                url=f"{BASE_URL}{self.project.label_row_metas[label_hash].data_hash}&{self.project.project_hash}/{frame}",
                img_id=get_image_identifier(du_hash, frame),
                class_id=class_id,
            )

            self.classification_labels.append(label_entry)

        def append_object_label(du_hash: str, frame: int, o: dict, width: int, height: int):
            class_id = self.get_object_class_id(o)
            if class_id is None:  # Ignore unwanted classes (defined by what is in `self.object_class_id_lookup`)
                return

            label_hash = self.lr_lookup[du_hash]
            label_entry = LabelEntry(
                identifier=f"{label_hash}_{du_hash}_{frame:05d}_{o['objectHash']}",
                url=f"{BASE_URL}{self.project.label_row_metas[label_hash].data_hash}&{self.project.project_hash}/{frame}",
                img_id=get_image_identifier(du_hash, frame),
                class_id=class_id,
            )

            if o["shape"] in BoxShapes:
                try:
                    bbox = BoundingBox.parse_obj(o.get("boundingBox"))
                except:
                    return  # Invalid bounding box object

                if o["shape"] == ObjectShape.ROTATABLE_BOUNDING_BOX:
                    label_entry.theta = bbox.theta

                label_entry.x1 = round(bbox.x * width, 2)
                label_entry.y1 = round(bbox.y * height, 2)
                label_entry.x2 = round((bbox.x + bbox.w) * width, 2)
                label_entry.y2 = round((bbox.y + bbox.h) * height, 2)
            elif o["shape"] == ObjectShape.POLYGON:
                points = polyobj_to_nparray(o, width=width, height=height)
                if points.size == 0:
                    return
                label_entry.x1, label_entry.y1 = points.min(axis=0)
                label_entry.x2, label_entry.y2 = points.max(axis=0)

                mask = points_to_mask(points, width=width, height=height)
                label_entry.rle = binary_mask_to_rle(mask)
            else:
                # Only supporting polygons and bounding boxes.
                return

            self.object_labels.append(label_entry)

        for lr in tqdm(self.project.label_rows.values(), desc="Preparing labels", leave=True):
            data_type = lr["data_type"]
            answers = lr["classification_answers"]
            for du_hash, du in lr["data_units"].items():
                height = int(du["height"])
                width = int(du["width"])

                if data_type in ["img_group", "image"]:
                    frame = int(du["data_sequence"])
                    for label in du["labels"].get("objects", []):
                        append_object_label(du_hash, frame, label, width, height)
                    for label in du["labels"].get("classifications", []):
                        append_classification_label(du_hash, frame, label, answers)
                else:  # Video
                    for fr, labels in du["labels"].items():
                        frame = int(fr)
                        for label in labels.get("objects", []):
                            append_object_label(du_hash, frame, label, width, height)
                        for label in labels.get("classifications", []):
                            append_classification_label(du_hash, frame, label, answers)

    def get_object_class_id(self, obj_dict) -> Optional[ClassID]:
        fh = obj_dict["featureHash"]
        if "objectHash" in obj_dict and fh in self.object_class_id_lookup:
            return self.object_class_id_lookup[fh]
        return None

    def get_classification_class_id(
        self, classification: LabelClassification, classification_answer: ClassificationAnswer
    ) -> Optional[ClassID]:
        if len(classification_answer.answers) == 0:
            return None

        key = FrameClassification(
            feature_hash=classification.featureHash,
            attribute_hash=classification_answer.featureHash,
            # NOTE: since we only support radion buttons, at this point we should have only one answer
            option_hash=classification_answer.answers[0].featureHash,
        )
        return self.classification_class_id_lookup.get(key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Do the matching computations
        ious, ground_truths_matched = precompute_MAP_features(
            self.predictions, self.object_labels + self.classification_labels
        )

        # 0. The predictions
        pred_df = pd.DataFrame(self.predictions)
        # TODO change bbox to coordinate columns

        pred_df["iou"] = ious
        pred_df.to_csv(self.storage_dir / PREDICTIONS_FILENAME)

        # 1. The labels
        df = pd.DataFrame(self.object_labels + self.classification_labels)
        # TODO change bbox to coordinate columns
        if "rle" in df:
            df["rle"] = df["rle"].map(
                lambda x: " ".join(x.reshape(-1).astype(str).tolist()) if isinstance(x, np.ndarray) else x
            )
        df.to_csv(self.storage_dir / LABELS_FILE)

        # 2. The GT matches
        with (self.storage_dir / GROUND_TRUTHS_MATCHED_FILE).open("w") as f:
            json.dump(ground_truths_matched, f)

        # 3. The class idx map
        class_index = {}
        for k, v in self.object_class_id_lookup.items():
            if k[0] == "_" or v in class_index:
                continue

            class_index[v] = {
                "featureHash": k,
                "name": self.object_lookup[k].name,
                "color": self.object_lookup[k].color,
            }

        for frame_classification, class_id in self.classification_class_id_lookup.items():
            if class_id in class_index or frame_classification not in self.classification_lookup:
                continue

            selected_option = self.classification_lookup[frame_classification]
            class_index[class_id] = {
                "featureHash": frame_classification.feature_hash,
                "attributeHash": frame_classification.attribute_hash,
                "optionHash": selected_option.feature_node_hash,
                "name": selected_option.label,
            }

        with (self.storage_dir / CLASS_INDEX_FILE).open("w") as f:
            json.dump(class_index, f)

    @staticmethod
    def __generate_hash() -> str:
        return b64encode(uuid4().bytes[:6]).decode("utf-8")

    def __get_unique_object_hash(self) -> str:
        object_hash = self.__generate_hash()
        while object_hash in self.uuids:
            object_hash = self.__generate_hash()
        self.uuids.add(object_hash)
        return object_hash

    def add_prediction(self, prediction: Prediction) -> None:
        """
        Add a prediction to en encord-active project.

        :param prediction: The `prediction` to write.
        """
        rle = None
        data_hash = prediction.data_hash
        label_hash = self.lr_lookup.get(data_hash)
        if not label_hash:
            logger.warning(f"Couldn't match data hash `{data_hash}` to any label row")
            return

        du = self.project.label_rows[label_hash]["data_units"][data_hash]

        width = int(du["width"])
        height = int(du["height"])

        if prediction.classification:
            class_id = self.classification_class_id_lookup.get(prediction.classification)
            ptype = PredictionType.FRAME
            x1, y1, x2, y2 = [None, None, None, None]
        elif prediction.object:
            class_id = self.object_class_id_lookup.get(prediction.object.feature_hash)
            if isinstance(prediction.object.data, np.ndarray):
                polygon = prediction.object.data
                ptype = PredictionType.POLYGON
                if isinstance(polygon, list):
                    polygon = np.array(polygon)

                if polygon.ndim != 2:
                    raise ValueError("Polygon argument should have just 2 dimensions: [h, w] or [N, 2]")

                if polygon.shape[1] != 2:  # Polygon is mask
                    np_mask = polygon
                    x1, y1, w, h = cv2.boundingRect(polygon)  # type: ignore
                else:  # Polygon is points
                    # Read image size from label row
                    if np.issubdtype(polygon.dtype, np.integer):
                        polygon = polygon.astype(float) / np.array([[width, height]])

                    np_mask = points_to_mask(polygon, width=width, height=height)  # type: ignore
                    x1, y1, w, h = cv2.boundingRect((polygon * np.array([[width, height]])).reshape(-1, 1, 2).astype(int))  # type: ignore
                x2, y2 = x1 + w, y1 + h
                rle = binary_mask_to_rle(np_mask)

            else:
                bbox = prediction.object.data
                ptype = PredictionType.BBOX
                x1 = bbox.x * width
                y1 = bbox.y * height
                x2 = (bbox.x + bbox.w) * width
                y2 = (bbox.y + bbox.h) * height

            ontology_object = self.object_lookup[prediction.object.feature_hash]
            if ontology_object.shape.value != ptype.value:
                raise ValueError(
                    f"You've passed a {ptype.value} but the provided class id is of type " f"{ontology_object.shape}"
                )
        else:
            raise ValueError("Prediction must have exactly one of `object` or `classification`")

        _frame = 0
        if not prediction.frame:  # Try to infer frame number from data hash.
            label_row = self.project.label_rows[label_hash]
            data_unit = label_row["data_units"][data_hash]
            if "data_sequence" in data_unit:
                _frame = int(data_unit["data_sequence"])

        object_hash = self.__get_unique_object_hash()

        if class_id is None:
            raise ValueError(
                f"`class_uid` didn't match any key in the "
                f"{'`custom_object_map`' if self.custom_map else 'project ontology'}.\n"
                f"Options are: [{', '.join(self.object_class_id_lookup.keys())}]"
            )

        self.predictions.append(
            PredictionEntry(
                identifier=f"{label_hash}_{data_hash}_{_frame:05d}_{object_hash}",
                url=f"{BASE_URL}{self.project.label_row_metas[self.lr_lookup[data_hash]].data_hash}&{self.project.project_hash}/{_frame}",
                img_id=get_image_identifier(data_hash, _frame),
                class_id=class_id,
                confidence=prediction.confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                rle=rle,
            )
        )
