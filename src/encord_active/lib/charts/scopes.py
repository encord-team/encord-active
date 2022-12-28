from enum import Enum


class PredictionMatchScope(Enum):
    TRUE_POSITIVES = "true positives"
    FALSE_POSITIVES = "false positives"
    FALSE_NEGATIVES = "false negatives"
