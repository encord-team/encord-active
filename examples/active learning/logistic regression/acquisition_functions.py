import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from encord_active.lib.metrics.acquisition_functions import (
    Entropy,
    LeastConfidence,
    Margin,
    Variance,
)

config = yaml.safe_load(Path("./../config.yaml").read_text())
logreg_model = pickle.loads(Path(config["logreg_model_path"]).read_bytes())


def get_predicted_class_probabilities(image) -> Optional[np.ndarray]:
    image_array = np.asarray(image).flatten()
    return logreg_model.predict_proba([image_array])


class LogRegEntropy(Entropy):
    def get_predicted_class_probabilities(self, image) -> Optional[np.ndarray]:
        return get_predicted_class_probabilities(image)


class LogRegLeastConfidence(LeastConfidence):
    def get_predicted_class_probabilities(self, image) -> Optional[np.ndarray]:
        return get_predicted_class_probabilities(image)


class LogRegMargin(Margin):
    def get_predicted_class_probabilities(self, image) -> Optional[np.ndarray]:
        return get_predicted_class_probabilities(image)


class LogRegVariance(Variance):
    def get_predicted_class_probabilities(self, image) -> Optional[np.ndarray]:
        return get_predicted_class_probabilities(image)


if __name__ == "__main__":
    from encord_active.lib.metrics.execute import execute_metrics

    execute_metrics(
        [LogRegEntropy(), LogRegLeastConfidence(), LogRegMargin(), LogRegVariance()],
        data_dir=Path(config["project_dir"]),
    )
