import DocCardList from "@theme/DocCardList";

# Importing Model Predictions

:::caution

Every time you run any of these importers, previously imported predictions will be overwritten!
Make sure to [version your projects][project-versioning] if you want to be able to go back to previous model iterations.

:::

If you aren't familiar with how to build lists of `Prediction` objects, please have a look at [this](../../import/import-predictions) workflow tutorial first.
It will show you how to construct predictions for bounding boxes, polygons, masks, and classifications.

With these predictions in hand, importing them is done as follows:

```python
from pathlib import Path

from encord_active.lib.db.predictions import Prediction
from encord_active.lib.model_predictions.importers import import_predictions
from encord_active.lib.project import Project

project_path = Path("/path/to/your/project/root")

predictions: List[Prediction] = ...  # Your list of predictions

import_predictions(Project(project_path), project_path, predictions)
```

## Other Options

There are also a couple of other options for importing predictions that are stored in common file formats already. They can be found here:

<DocCardList />

[project-versioning]: ../../user-guide/versioning
