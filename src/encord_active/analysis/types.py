from typing import Dict, List, TypedDict


class ObjectMetadata(TypedDict):
    pass


class ClassificationMetadata:
    pass


class ObjectAndClassifications(TypedDict):
    objects: List[ObjectMetadata]
    classifications: List[ClassificationMetadata]


class LabelMetadata(TypedDict):
    id: int
    label_hash: str
    user_hash: str
    data_hash: str
    project_hash: str
    label_status: int
    labels: Dict[str, ObjectAndClassifications]
