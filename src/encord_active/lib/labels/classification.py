import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Set, Union

from encord.objects.common import NestableOption, RadioAttribute
from encord.objects.ontology_structure import Classification, OntologyStructure
from encord.orm.label_row import LabelRow

from encord_active.lib.common.time import get_timestamp


@dataclass
class LabelClassification:
    name: str
    value: str
    createdAt: str
    createdBy: str
    confidence: float
    featureHash: str
    manualAnnotation: bool


@dataclass
class Answer:
    name: str
    value: str
    featureHash: str


@dataclass
class ClassificationAnswer:
    name: str
    value: str
    featureHash: str
    manualAnnotation: bool
    answers: List[Answer] = field(default_factory=list)


def build_label_classification(classification: Classification) -> LabelClassification:
    attr = classification.attributes[0]
    return LabelClassification(
        name=attr.name,
        value=attr.name.lower(),
        createdAt=get_timestamp(),
        createdBy="robot@cord.tech",
        confidence=1.0,
        featureHash=classification.feature_node_hash,
        manualAnnotation=True,
    )


def build_classification_answer(
    label_classification: LabelClassification, option: NestableOption, attribute: RadioAttribute
):
    return ClassificationAnswer(
        name=label_classification.name,
        value=label_classification.value,
        featureHash=attribute.feature_node_hash,
        answers=[Answer(name=option.label, value=option.value, featureHash=option.feature_node_hash)],
        manualAnnotation=True,
    )


def update_label_row_with_classification(
    label_row: LabelRow, classification: Classification, image_class: str
) -> LabelRow:
    attribute = classification.attributes[0]
    classification_hash = str(uuid.uuid4())[:8]
    data_hash = list(label_row["data_units"].keys())[0]
    label_classification = build_label_classification(classification)

    label_row["data_units"][data_hash]["labels"]["classifications"] = [
        {**asdict(label_classification), "classificationHash": classification_hash}
    ]

    if not isinstance(attribute, RadioAttribute):
        raise ValueError("Classification attribute should be radio attribute")

    option, *_ = [option for option in attribute.options if option.value == image_class]

    label_row["classification_answers"] = {
        classification_hash: {
            "classificationHash": classification_hash,
            "classifications": [asdict(build_classification_answer(label_classification, option, attribute))],
        }
    }

    return label_row


def create_ontology_structure(classnames: Union[Set[str], List[str]]):
    ontology = OntologyStructure()
    classification = ontology.add_classification()
    attribute = classification.add_attribute(RadioAttribute, "Classification", required=True)
    for name in classnames:
        attribute.add_option(label=name)

    return ontology
