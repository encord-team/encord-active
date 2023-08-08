import enum
from dataclasses import dataclass
from typing import Dict, Optional


class EnumType(enum.Enum):
    ONTOLOGY = "ontology"
    """Ontology well-known enum"""
    ENUM = "enum"
    """Simple enum with fixed values"""


@dataclass
class EnumDefinition:
    enum_type: EnumType
    title: str
    values: Optional[Dict[str, str]] = None


class AnnotationType(enum.IntEnum):
    CLASSIFICATION = 0  # "classification"
    BOUNDING_BOX = 1  # "bounding_box"
    ROTATABLE_BOUNDING_BOX = 2  # "rotatable_bounding_box"
    POINT = 3  # "point"
    POLYLINE = 4  # "polyline"
    POLYGON = 5  # "polygon"
    SKELETON = 6  # "skeleton"
    BITMASK = 7  # "bitmask"


_ANNOTATION_TYPE_LOOKUP: Dict[str, AnnotationType] = {
    "classification": AnnotationType.CLASSIFICATION,
    "bounding_box": AnnotationType.BOUNDING_BOX,
    "rotatable_bounding_box": AnnotationType.ROTATABLE_BOUNDING_BOX,
    "point": AnnotationType.POINT,
    "polyline": AnnotationType.POLYLINE,
    "polygon": AnnotationType.POLYGON,
    "skeleton": AnnotationType.SKELETON,
    "bitmask": AnnotationType.BITMASK,
}


def annotation_type_from_str(value: str) -> AnnotationType:
    return _ANNOTATION_TYPE_LOOKUP[value]


AnnotationTypeMaxValue: int = int(AnnotationType.BITMASK)

DataEnums: Dict[str, EnumDefinition] = {}
AnnotationEnums: Dict[str, EnumDefinition] = {
    "feature_hash": EnumDefinition(
        enum_type=EnumType.ONTOLOGY,
        title="Label Class",
        values=None,
    ),
    "annotation_type": EnumDefinition(
        enum_type=EnumType.ENUM,
        title="Annotation Type",
        values={
            str(annotation_type.value): annotation_type.name.replace("_", "").title()
            for annotation_type in AnnotationType
        },
    ),
    "annotation_manual": EnumDefinition(
        enum_type=EnumType.ENUM,
        title="Manual Annotation",
        values={
            "False": "Automated Annotation",
            "True": "Manual Annotation",
        },
    ),
}
