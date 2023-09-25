from enum import Enum


class RouteTag(Enum):
    PROJECT = "project"
    PREDICTION = "prediction"


class AnalysisDomain(Enum):
    Data = "data"
    Annotation = "annotation"
