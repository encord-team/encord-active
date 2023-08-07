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


class AnnotationType(enum.Enum):
    CLASSIFICATION = "classification"
    BOUNDING_BOX = "bounding_box"
    ROT_BOUNDING_BOX = "rotatable_bounding_box"
    POINT = "point"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    SKELETON = "skeleton"
    BITMASK = "bitmask"


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
            annotation_type.value: annotation_type.name.replace("_", "").title() for annotation_type in AnnotationType
        },
    ),
    "annotation_manual": EnumDefinition(
        enum_type=EnumType.ENUM,
        title="Manual Annotation",
        values={
            "0": "Automated Annotation",
            "1": "Manual Annotation",
        },
    ),
}
