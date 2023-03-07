from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from encord_active.lib.common.iterator import DatasetIterator

config = yaml.safe_load(Path("./config.yaml").read_text())
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train phase
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# get predictions
y_pred = logreg.predict(X_test)

# show scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print("Classes:", logreg.classes_)
print("Accuracy:", accuracy)
print("precision", list(precision))
print("recall", list(recall))
print("f1", list(f1))
