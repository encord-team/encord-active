import logging
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as torch_transforms
from encord.project_ontology.classification_type import ClassificationType
from encord.project_ontology.object_type import ObjectShape
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from torchvision.models.feature_extraction import create_feature_extractor

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import get_bbox_from_encord_label_object

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_transforms() -> Tuple[nn.Module, nn.Module]:
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights).to(DEVICE)
    embedding_extractor = create_feature_extractor(model, return_nodes={"avgpool": "my_avgpool"})
    for p in embedding_extractor.parameters():
        p.requires_grad = False
    embedding_extractor.eval()
    return embedding_extractor, weights.transforms()


def adjust_image_channels(image: torch.Tensor) -> torch.Tensor:
    if image.shape[0] == 4:
        image = image[:3]
    elif image.shape[0] < 3:
        image = image.repeat(3, 1, 1)

    return image


def image_path_to_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path.as_posix())
    transform = torch_transforms.ToTensor()
    image = transform(image)

    image = adjust_image_channels(image)

    return image


def assemble_object_batch(data_unit: dict, img_path: Path, transforms: Optional[nn.Module]):
    if transforms is None:
        transforms = torch.nn.Sequential()

    image = image_path_to_tensor(img_path)
    img_batch: List[torch.Tensor] = []

    for obj in data_unit["labels"].get("objects", []):
        if obj["shape"] in [ObjectShape.POLYGON.value, ObjectShape.BOUNDING_BOX.value]:
            try:
                out = get_bbox_from_encord_label_object(
                    obj,
                    image.shape[2],
                    image.shape[1],
                )

                if out is None:
                    continue
                x, y, w, h = out

                img_patch = image[:, y : y + h, x : x + w]
                img_batch.append(transforms(img_patch))
            except Exception as e:
                logger.warning(f"Error with object {obj['objectHash']}: {e}")
                continue
    return torch.stack(img_batch).to(DEVICE) if len(img_batch) > 0 else None


@torch.inference_mode()
def generate_cnn_embeddings(iterator: Iterator, filepath: str) -> None:
    start = time.perf_counter()
    feature_extractor, transforms = get_model_and_transforms()

    collections = []
    for data_unit, img_pth in iterator.iterate(desc="Embedding object data."):
        if img_pth is None:
            continue

        batches = assemble_object_batch(data_unit, img_pth, transforms=transforms)
        if batches is None:
            continue

        embeddings = feature_extractor(batches)["my_avgpool"]
        embeddings_torch = torch.flatten(embeddings, start_dim=1).cpu().detach().numpy()

        for obj, emb in zip(data_unit["labels"]["objects"], embeddings_torch):
            if obj["shape"] in [ObjectShape.POLYGON.value, ObjectShape.BOUNDING_BOX.value]:

                last_edited_by = obj["lastEditedBy"] if "lastEditedBy" in obj.keys() else obj["createdBy"]

                entry = {
                    "label_row": iterator.label_hash,
                    "data_unit": data_unit["data_hash"],
                    "frame": iterator.frame,
                    "objectHash": obj["objectHash"],
                    "lastEditedBy": last_edited_by,
                    "featureHash": obj["featureHash"],
                    "name": obj["name"],
                    "dataset_title": iterator.dataset_title,
                    "embedding": emb,
                }

                collections.append(entry)

    with open(filepath, "wb") as f:
        pickle.dump(collections, f)

    logger.info(
        f"Generating {len(iterator)} embeddings took {str(time.perf_counter() - start)} seconds",
    )


@torch.inference_mode()
def generate_cnn_classification_embeddings(iterator: Iterator, filepath: str) -> None:
    ontology_class_hash_to_index: dict[str, dict] = {}
    ontology_class_hash_to_question_hash: dict[str, str] = {}

    # TODO: This only evaluates immediate classes, not nested ones
    # TODO: Look into other types of classifications apart from radio buttons
    for class_label in iterator.project.ontology["classifications"]:
        class_question = class_label["attributes"][0]
        if class_question["type"] == ClassificationType.RADIO.value:
            ontology_class_hash = class_label["featureNodeHash"]
            ontology_class_hash_to_index[ontology_class_hash] = {}
            ontology_class_hash_to_question_hash[ontology_class_hash] = class_question["featureNodeHash"]
            for index, option in enumerate(class_question["options"]):
                ontology_class_hash_to_index[ontology_class_hash][option["featureNodeHash"]] = index

    start = time.perf_counter()
    feature_extractor, transforms = get_model_and_transforms()

    collections = []
    for data_unit, img_pth in iterator.iterate(desc="Embedding image data."):
        if not img_pth:
            continue
        temp_entry = {}
        temp_entry["label_row"] = iterator.label_hash
        temp_entry["data_unit"] = data_unit["data_hash"]
        temp_entry["frame"] = str(iterator.frame)
        temp_entry["url"] = data_unit["data_link"]

        temp_classification_hash = {}
        classification_answers = iterator.label_rows[iterator.label_hash]["classification_answers"]
        for classification in data_unit["labels"].get("classifications", []):

            classification_hash = classification["classificationHash"]
            ontology_class_hash = classification["featureHash"]
            if ontology_class_hash in ontology_class_hash_to_index.keys():
                for classification_answer in classification_answers[classification_hash]["classifications"]:
                    if (
                        classification_answer["featureHash"]
                        == ontology_class_hash_to_question_hash[ontology_class_hash]
                    ):
                        temp_classification_hash[ontology_class_hash] = {
                            "answer_featureHash": classification_answer["answers"][0]["featureHash"],
                            "answer_name": classification_answer["answers"][0]["name"],
                            "annotator": classification["createdBy"],
                        }

        temp_entry["classification_answers"] = temp_classification_hash  # type: ignore[assignment]

        image = image_path_to_tensor(img_pth)
        transformed_image = transforms(image).unsqueeze(0)
        embedding = feature_extractor(transformed_image.to(DEVICE))["my_avgpool"]
        embedding = torch.flatten(embedding).cpu().detach().numpy()

        temp_entry["embedding"] = embedding

        collections.append(temp_entry)

    with open(filepath, "wb") as f:
        pickle.dump(collections, f)

    logger.info(
        f"Generating {len(iterator)} embeddings took {str(time.perf_counter() - start)} seconds",
    )


def get_cnn_embeddings(iterator: Iterator, embedding_type: str = "objects", *, force: bool = False) -> list:
    target_folder = os.path.join(iterator.cache_dir, "embeddings")
    if embedding_type == "objects":
        embedding_file = "cnn_objects"
    elif embedding_type == "classifications":
        embedding_file = "cnn_classifications"
    else:
        raise Exception(f"Undefined embedding type '{embedding_type}' for get_cnn_embeddings method")

    embedding_path = os.path.join(target_folder, f"{embedding_file}.pkl")

    if force:
        logger.info("Regenerating CNN embeddings...")
        if embedding_type == "objects":
            generate_cnn_embeddings(iterator, filepath=embedding_path)
        elif embedding_type == "classifications":
            generate_cnn_classification_embeddings(iterator, filepath=embedding_path)

        with open(embedding_path, "rb") as f:
            cnn_embeddings = pickle.load(f)
        logger.info("Done!")
    else:
        try:
            with open(embedding_path, "rb") as f:
                cnn_embeddings = pickle.load(f)
        except FileNotFoundError:
            logger.info(f"{embedding_path} not found. Generating embeddings...")
            if embedding_type == "objects":
                generate_cnn_embeddings(iterator, filepath=embedding_path)
            elif embedding_type == "classifications":
                generate_cnn_classification_embeddings(iterator, filepath=embedding_path)
            with open(embedding_path, "rb") as f:
                cnn_embeddings = pickle.load(f)
            logger.info("Done!")

    return cnn_embeddings
