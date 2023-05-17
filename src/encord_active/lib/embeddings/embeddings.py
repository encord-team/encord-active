import pickle
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from encord.objects.common import PropertyType
from encord.project_ontology.object_type import ObjectShape
from loguru import logger
from PIL import Image

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import get_bbox_from_encord_label_object
from encord_active.lib.embeddings.models.clip_embedder import CLIPEmbedder
from encord_active.lib.embeddings.models.embedder_model import ImageEmbedder
from encord_active.lib.embeddings.utils import ClassificationAnswer, LabelEmbedding
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.project.project_file_structure import ProjectFileStructure


def get_default_embedder() -> ImageEmbedder:
    return CLIPEmbedder()


def assemble_object_batch(data_unit: dict, img_path: Path) -> List[Image.Image]:
    try:
        image = np.asarray(Image.open(img_path).convert("RGB"))
    except OSError:
        logger.warning(f"Image with path {img_path} seems to be broken. Skipping.")
        return []

    img_h, img_w, *_ = image.shape
    img_batch: List[Image.Image] = []

    for obj in data_unit["labels"].get("objects", []):
        if obj["shape"] in [
            ObjectShape.POLYGON.value,
            ObjectShape.BOUNDING_BOX.value,
            ObjectShape.ROTATABLE_BOUNDING_BOX.value,
        ]:
            try:
                out = get_bbox_from_encord_label_object(
                    obj,
                    w=img_w,
                    h=img_h,
                )

                if out is None:
                    continue
                x, y, w, h = out

                img_patch = image[y : y + h, x : x + w]
                img_batch.append(Image.fromarray(img_patch))
            except Exception as e:
                logger.warning(f"Error with object {obj['objectHash']}: {e}")
                continue
    return img_batch


@torch.inference_mode()
def generate_image_embeddings(
    iterator: Iterator, feature_extractor: Optional[ImageEmbedder] = None, batch_size=100
) -> List[LabelEmbedding]:
    start = time.perf_counter()
    if feature_extractor is None:
        feature_extractor = get_default_embedder()

    raw_embeddings: list[np.ndarray] = []
    batch = []
    skip: set[int] = set()
    for i, (data_unit, img_pth) in enumerate(iterator.iterate(desc="Embedding image data.")):
        if img_pth is None:
            skip.add(i)
            continue
        try:
            batch.append(Image.open(img_pth).convert("RGB"))
        except OSError:
            logger.warning(f"Image with path {img_pth} seems to be broken. Skipping.")
            skip.add(i)
            continue

        if len(batch) >= batch_size:
            raw_embeddings.append(feature_extractor.embed_images(batch))
            batch = []

    if batch:
        raw_embeddings.append(feature_extractor.embed_images(batch))

    if len(raw_embeddings) > 1:
        raw_np_embeddings = np.concatenate(raw_embeddings)
    else:
        raw_np_embeddings = raw_embeddings[0]

    collections: List[LabelEmbedding] = []
    offset = 0
    for i, (data_unit, img_pth) in enumerate(iterator.iterate(desc="Storing embeddings.")):
        if i in skip:
            offset += 1
            continue

        embedding = raw_np_embeddings[i - offset]
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

    logger.info(
        f"Generating {len(iterator)} embeddings took {str(time.perf_counter() - start)} seconds",
    )

    return collections


@torch.inference_mode()
def generate_object_embeddings(
    iterator: Iterator, feature_extractor: Optional[ImageEmbedder] = None
) -> List[LabelEmbedding]:
    start = time.perf_counter()
    if feature_extractor is None:
        feature_extractor = get_default_embedder()

    collections: List[LabelEmbedding] = []
    for data_unit, img_pth in iterator.iterate(desc="Embedding object data."):
        if img_pth is None:
            continue

        batch = assemble_object_batch(data_unit, img_pth)
        if not batch:
            continue

        embeddings = feature_extractor.embed_images(batch)
        for obj, emb in zip(data_unit["labels"].get("objects", []), embeddings):
            if obj["shape"] not in [
                ObjectShape.POLYGON.value,
                ObjectShape.BOUNDING_BOX.value,
                ObjectShape.ROTATABLE_BOUNDING_BOX.value,
            ]:
                continue

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

    logger.info(
        f"Generating {len(iterator)} embeddings took {str(time.perf_counter() - start)} seconds",
    )

    return collections


@torch.inference_mode()
def generate_classification_embeddings(
    iterator: Iterator, feature_extractor: Optional[ImageEmbedder]
) -> List[LabelEmbedding]:
    image_collections = get_embeddings(iterator, embedding_type=EmbeddingType.IMAGE)

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
            for index, option in enumerate(class_question.options):  # type: ignore
                ontology_class_hash_to_index[ontology_class_hash][option.feature_node_hash] = index

    start = time.perf_counter()
    if feature_extractor is None:
        feature_extractor = get_default_embedder()

    collections = []
    for data_unit, img_pth in iterator.iterate(desc="Embedding classification data."):
        if img_pth is None:
            continue

        matching_image_collections = [
            collection
            for collection in image_collections
            if collection["data_unit"] == data_unit["data_hash"]
            and collection["label_row"] == iterator.label_hash
            and collection["frame"] == iterator.frame
        ]

        if not matching_image_collections:
            try:
                image = Image.open(img_pth).convert("RGB")
            except OSError:
                logger.warning(f"Image with path {img_pth} seems to be broken. Skipping.")
                continue

            embedding = feature_extractor.embed_image(image)
        else:
            embedding = matching_image_collections[0]["embedding"]

        if embedding is None:
            continue

        classification_answers = iterator.label_rows[iterator.label_hash]["classification_answers"]
        for classification in data_unit["labels"].get("classifications", []):
            last_edited_by = (
                classification["lastEditedBy"]
                if "lastEditedBy" in classification.keys()
                else classification["createdBy"]
            )
            classification_hash = classification["classificationHash"]
            ontology_class_hash = classification["featureHash"]

            if ontology_class_hash not in ontology_class_hash_to_question_hash:
                continue

            answers: List[ClassificationAnswer] = []
            if ontology_class_hash in ontology_class_hash_to_index.keys() and classification_answers:
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
            # NOTE: since we only support one one classification for now
            identified_answers = answers[0] if len(answers) else None

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
                classification_answers=identified_answers,
            )
            collections.append(entry)

    logger.info(
        f"Generating {len(iterator)} embeddings took {str(time.perf_counter() - start)} seconds",
    )

    return collections


def get_embeddings(iterator: Iterator, embedding_type: EmbeddingType, *, force: bool = False) -> List[LabelEmbedding]:
    if embedding_type not in [EmbeddingType.CLASSIFICATION, EmbeddingType.IMAGE, EmbeddingType.OBJECT]:
        raise Exception(f"Undefined embedding type '{embedding_type}' for get_embeddings method")

    pfs = ProjectFileStructure(iterator.cache_dir)
    embedding_path = pfs.get_embeddings_file(embedding_type)

    if force:
        logger.info("Regenerating CNN embeddings...")
        embeddings = generate_embeddings(iterator, embedding_type, embedding_path)
    else:
        try:
            with open(embedding_path, "rb") as f:
                embeddings = pickle.load(f)
        except FileNotFoundError:
            logger.info(f"{embedding_path} not found. Generating embeddings...")
            embeddings = generate_embeddings(iterator, embedding_type, embedding_path)

    return embeddings


def generate_embeddings(
    iterator: Iterator, embedding_type: EmbeddingType, target: Path, feature_extractor: Optional[ImageEmbedder] = None
):
    if embedding_type == EmbeddingType.IMAGE:
        embeddings = generate_image_embeddings(iterator, feature_extractor=feature_extractor)
    elif embedding_type == EmbeddingType.OBJECT:
        embeddings = generate_object_embeddings(iterator, feature_extractor=feature_extractor)
    elif embedding_type == EmbeddingType.CLASSIFICATION:
        embeddings = generate_classification_embeddings(iterator, feature_extractor=feature_extractor)
    else:
        raise ValueError(f"Unsupported embedding type {embedding_type}")

    if embeddings:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pickle.dumps(embeddings))

    logger.info("Done!")

    return embeddings
