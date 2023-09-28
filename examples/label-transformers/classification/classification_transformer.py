from pathlib import Path
from typing import List

from encord_active.public.label_transformer import (
    ClassificationLabel,
    DataLabel,
    LabelTransformer,
)


class ClassificationTransformer(LabelTransformer):
    def from_custom_labels(self, _, data_files: List[Path]) -> List[DataLabel]:
        return [DataLabel(f, ClassificationLabel(class_=f.parent.name)) for f in data_files]
