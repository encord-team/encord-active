import pickle
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from encord_active.lib.common.iterator import DatasetIterator

config = yaml.safe_load(Path("./../config.yaml").read_text())
project_dir = Path(config["project_dir"])

iterator = DatasetIterator(project_dir)

X = []
y = []
for data_unit, img_pth in iterator.iterate():
    filtered_class = [_class for _class in data_unit["labels"]["classifications"] if _class["name"] == "Classification"]
    if len(filtered_class) == 0:
        continue
    label_row = iterator.label_rows[iterator.label_hash]

    class_hash = filtered_class[0]["classificationHash"]
    class_label = label_row["classification_answers"][class_hash]["classifications"][0]["answers"][0]["name"]
    image_array = np.asarray(Image.open(img_pth))

    X.append(image_array.flatten())
    y.append(class_label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train phase
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# save model to disk
logreg_model_path = Path(config["logreg_model_path"])
logreg_model_path.write_bytes(pickle.dumps(logreg))
