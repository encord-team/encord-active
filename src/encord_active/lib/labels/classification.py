import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, List, Optional, Set, Union

from encord.objects.classification import Classification
from encord.objects.common import NestableOption, RadioAttribute, TextAttribute
from encord.objects.ontology_structure import OntologyStructure
from encord.orm.label_row import LabelRow
from pydantic import BaseModel

from encord_active.lib.common.time import get_timestamp


# copy from encord but as a string enum
class ClassificationType(str, Enum):
    RADIO = "radio"
    TEXT = "text"
    CHECKLIST = "checklist"


@dataclass
class LabelClassification:
    name: str
    value: str
    createdAt: str
    createdBy: str
    confidence: float
    featureHash: str
    manualAnnotation: bool
    classificationHash: str
    reviews: List[Any] = field(default_factory=list)
    lastEditedAt: Optional[str] = None
    lastEditedBy: Optional[str] = None


class Answer(BaseModel):
    name: str
    value: str
    featureHash: str


class ClassificationAnswer(BaseModel):
    name: str
    value: str
    featureHash: str
    manualAnnotation: bool
    answers: Union[List[Answer], str]
