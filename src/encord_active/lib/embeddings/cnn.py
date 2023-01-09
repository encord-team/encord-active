import logging
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as torch_transforms
from encord.objects.common import PropertyType
from encord.project_ontology.object_type import ObjectShape
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from torchvision.models.feature_extraction import create_feature_extractor

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import get_bbox_from_encord_label_object
from encord_active.lib.embeddings.utils import (
    EMBEDDING_TYPE_TO_FILENAME,
    ClassificationAnswer,
    LabelEmbedding,
)
from encord_active.lib.metrics.metric import EmbeddingType

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
def generate_cnn_image_embeddings(iterator: Iterator, filepath: str) -> None:
    start = time.perf_counter()
    feature_extractor, transforms = get_model_and_transforms()

    collections: List[LabelEmbedding] = []
    for data_unit, img_pth in iterator.iterate(desc="Embedding image data."):
        if img_pth is None:
            continue

        image = image_path_to_tensor(img_pth)
        transformed_image = transforms(image).unsqueeze(0)
        embedding = feature_extractor(transformed_image.to(DEVICE))["my_avgpool"]
        embedding = torch.flatten(embedding).cpu().detach().numpy()

        entry = LabelEmbedding(
            url=data_unit["data_link"],
            label_row=iterator.label_hash,
            data_unit=data_unit["data_hash"],
            frame=iterator.frame,
            labelHash=None,
            lastEditedBy=None,
            featureHash=None,
            name=None,
            dataset_title=iterator.dataset_title,
            embedding=embedding,
            classification_answers=None,
        )
        collections.append(entry)

    with open(filepath, "wb") as f:
        pickle.dump(collections, f)

    logger.info(
        f"Generating {len(iterator)} embeddings took {str(time.perf_counter() - start)} seconds",
    )


@torch.inference_mode()
def generate_cnn_object_embeddings(iterator: Iterator, filepath: str) -> None:
    start = time.perf_counter()
    feature_extractor, transforms = get_model_and_transforms()

    collections: List[LabelEmbedding] = []
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

                entry = LabelEmbedding(
                    url=data_unit["data_link"],
                    label_row=iterator.label_hash,
                    data_unit=data_unit["data_hash"],
                    frame=iterator.frame,
                    labelHash=obj["objectHash"],
                    lastEditedBy=last_edited_by,
                    featureHash=obj["featureHash"],
                    name=obj["name"],
                    dataset_title=iterator.dataset_title,
                    embedding=emb,
                    classification_answers=None,
                )

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
    for class_label in iterator.project.ontology.classifications:
        class_question = class_label.attributes[0]
        if class_question.get_property_type() == PropertyType.RADIO:
            ontology_class_hash = class_label.feature_node_hash
            ontology_class_hash_to_index[ontology_class_hash] = {}
            ontology_class_hash_to_question_hash[ontology_class_hash] = class_question.feature_node_hash
            for index, option in enumerate(class_question.options):
                ontology_class_hash_to_index[ontology_class_hash][option.feature_node_hash] = index

    start = time.perf_counter()
    feature_extractor, transforms = get_model_and_transforms()

    collections = []
    for data_unit, img_pth in iterator.iterate(desc="Embedding classification data."):
        if not img_pth:
            continue

        image = image_path_to_tensor(img_pth)
        transformed_image = transforms(image).unsqueeze(0)
        embedding = feature_extractor(transformed_image.to(DEVICE))["my_avgpool"]
        embedding = torch.flatten(embedding).cpu().detach().numpy()

        classification_answers = iterator.label_rows[iterator.label_hash]["classification_answers"]
        for classification in data_unit["labels"].get("classifications", []):

            last_edited_by = (
                classification["lastEditedBy"]
                if "lastEditedBy" in classification.keys()
                else classification["createdBy"]
            )
            classification_hash = classification["classificationHash"]
            ontology_class_hash = classification["featureHash"]

            answers: List[ClassificationAnswer] = []
            if ontology_class_hash in ontology_class_hash_to_index.keys():
                for classification_answer in classification_answers[classification_hash]["classifications"]:
                    if (
                        classification_answer["featureHash"]
                        == ontology_class_hash_to_question_hash[ontology_class_hash]
                    ):
                        answers.append(
                            ClassificationAnswer(
                                answer_featureHash=classification_answer["answers"][0]["featureHash"],
                                answer_name=classification_answer["answers"][0]["name"],
                                annotator=classification["createdBy"],
                            )
                        )

            entry = LabelEmbedding(
                url=data_unit["data_link"],
                label_row=iterator.label_hash,
                data_unit=data_unit["data_hash"],
                frame=iterator.frame,
                labelHash=classification_hash,
                lastEditedBy=last_edited_by,
                featureHash=ontology_class_hash,
                name=classification["name"],
                dataset_title=iterator.dataset_title,
                embedding=embedding,
                classification_answers=answers[0],
            )
            collections.append(entry)

    with open(filepath, "wb") as f:
        pickle.dump(collections, f)

    logger.info(
        f"Generating {len(iterator)} embeddings took {str(time.perf_counter() - start)} seconds",
    )


def get_cnn_embeddings(iterator: Iterator, embedding_type: EmbeddingType, *, force: bool = False) -> list:
    if embedding_type not in [EmbeddingType.CLASSIFICATION, EmbeddingType.IMAGE, EmbeddingType.OBJECT]:
        raise Exception(f"Undefined embedding type '{embedding_type}' for get_cnn_embeddings method")

    target_folder = os.path.join(iterator.cache_dir, "embeddings")
    embedding_path = os.path.join(target_folder, f"{EMBEDDING_TYPE_TO_FILENAME[embedding_type]}")

    if force:
        logger.info("Regenerating CNN embeddings...")
        cnn_embeddings = generate_cnn_embeddings(iterator, embedding_type, embedding_path)
    else:
        try:
            with open(embedding_path, "rb") as f:
                cnn_embeddings = pickle.load(f)
        except FileNotFoundError:
            logger.info(f"{embedding_path} not found. Generating embeddings...")
            cnn_embeddings = generate_cnn_embeddings(iterator, embedding_type, embedding_path)

    return cnn_embeddings


def generate_cnn_embeddings(iterator: Iterator, embedding_type: EmbeddingType, target: str):
    if embedding_type == EmbeddingType.IMAGE:
        generate_cnn_image_embeddings(iterator, filepath=target)
    elif embedding_type == EmbeddingType.OBJECT:
        generate_cnn_object_embeddings(iterator, filepath=target)
    elif embedding_type == EmbeddingType.CLASSIFICATION:
        generate_cnn_classification_embeddings(iterator, filepath=target)

    with open(target, "rb") as f:
        cnn_embeddings = pickle.load(f)
    logger.info("Done!")

    return cnn_embeddings
