import enum
from dataclasses import dataclass
from typing import Dict, Optional


class EnumType(enum.Enum):
    ONTOLOGY = "ontology"
    """Ontology well-known enum"""
    USER_EMAIL = "user_email"
    """User Email well-known enum"""
    TAGS = "tags"
    """Tag search support"""
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

    @classmethod
    def _missing_(cls, value: object) -> "AnnotationType":
        if isinstance(value, str):
            v_upper = value.upper()
            for v in cls:
                if v.name == v_upper:
                    return v
        raise ValueError(f"Unknown AnnotationType: {value}")

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return self.__str__()

    # OpenAPI schema
    @classmethod
    def __modify_schema__(cls, schema: Dict[str, list]) -> Dict[str, list]:
        values = [c for c in cls]
        schema["enum"] = [v.value for v in values]
        schema["x-enum-varnames"] = [v.name for v in values]
        return schema


AnnotationTypeMaxValue: int = int(AnnotationType.BITMASK)


class DataType(enum.IntEnum):
    IMAGE = 0
    IMG_GROUP = 1
    IMG_SEQUENCE = 2
    VIDEO = 3

    @classmethod
    def _missing_(cls, value: object) -> "DataType":
        if isinstance(value, str):
            v_upper = value.upper()
            for v in cls:
                if v.name == v_upper:
                    return v
        raise ValueError(f"Unknown DataType: {value}")

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return self.__str__()

    def __int__(self):
        return self.value

    # OpenAPI schema
    @classmethod
    def __modify_schema__(cls, schema: Dict[str, list]) -> Dict[str, list]:
        values = [c for c in cls]
        schema["enum"] = [v.value for v in values]
        schema["x-enum-varnames"] = [v.name for v in values]
        return schema


DataTypeMaxValue: int = int(DataType.VIDEO)

DataEnums: Dict[str, EnumDefinition] = {
    "data_type": EnumDefinition(
        enum_type=EnumType.ENUM,
        title="Data Type",
        values={
            str(data_type.value): data_type.name.replace("IMG_", "IMAGE_").replace("_", " ").title()
            for data_type in DataType
        },
    ),
    "tags": EnumDefinition(
        enum_type=EnumType.TAGS,
        title="Data Tags",
        values=None,
    ),
}
AnnotationEnums: Dict[str, EnumDefinition] = {
    "feature_hash": EnumDefinition(
        enum_type=EnumType.ONTOLOGY,
        title="Annotation Class",
        values=None,
    ),
    "annotation_type": EnumDefinition(
        enum_type=EnumType.ENUM,
        title="Annotation Type",
        values={
            str(annotation_type.value): annotation_type.name.replace("_", " ").title()
            for annotation_type in AnnotationType
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
    "annotation_invalid": EnumDefinition(
        enum_type=EnumType.ENUM,
        title="Annotation Invalid",
        values={
            "0": "Well-formed Annotation",
            "1": "Invalid Annotation",
        },
    ),
    "annotation_user_id": EnumDefinition(
        enum_type=EnumType.USER_EMAIL,
        title="Annotator",
        values=None,
    ),
    "tags": EnumDefinition(
        enum_type=EnumType.TAGS,
        title="Annotation Tags",
        values=None,
    ),
}
