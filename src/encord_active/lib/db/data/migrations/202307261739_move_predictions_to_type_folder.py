import os

from encord_active.lib.model_predictions.reader import (
    check_model_prediction_availability,
)
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.project import ProjectFileStructure


def up(pfs: ProjectFileStructure) -> None:
    if not check_model_prediction_availability(pfs.predictions):
        return

    prediction_type = (
        MainPredictionType.OBJECT
        if (pfs.predictions / "ground_truths_matched.json").exists()
        else MainPredictionType.CLASSIFICATION
    )
    out_dir = pfs.predictions / prediction_type.value

    for child in pfs.predictions.glob("*"):
        os.renames(child, out_dir / child.name)
