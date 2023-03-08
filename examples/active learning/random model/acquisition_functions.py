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


def generate_uniform_random_probabilities(size):
    return np.random.uniform(size=size)


def get_predicted_class_probabilities(image) -> Optional[np.ndarray]:
    return generate_uniform_random_probabilities(size=(1, 4))


class URandEntropy(Entropy):
    def __init__(self):
        super().__init__()
        self.metadata.title += " (Uniform Random)"

    def get_predicted_class_probabilities(self, image) -> Optional[np.ndarray]:
        return get_predicted_class_probabilities(image)


class URandLeastConfidence(LeastConfidence):
    def __init__(self):
        super().__init__()
        self.metadata.title += " (Uniform Random)"

    def get_predicted_class_probabilities(self, image) -> Optional[np.ndarray]:
        return get_predicted_class_probabilities(image)


class URandMargin(Margin):
    def __init__(self):
        super().__init__()
        self.metadata.title += " (Uniform Random)"

    def get_predicted_class_probabilities(self, image) -> Optional[np.ndarray]:
        return get_predicted_class_probabilities(image)


class URandVariance(Variance):
    def __init__(self):
        super().__init__()
        self.metadata.title += " (Uniform Random)"

    def get_predicted_class_probabilities(self, image) -> Optional[np.ndarray]:
        return get_predicted_class_probabilities(image)


if __name__ == "__main__":
    from encord_active.lib.metrics.execute import execute_metrics

    config = yaml.safe_load(Path("./../config.yaml").read_text())

    execute_metrics(
        [URandEntropy(), URandLeastConfidence(), URandMargin(), URandVariance()],
        data_dir=Path(config["project_dir"]),
    )
