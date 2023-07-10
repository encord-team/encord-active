import DocCardList from "@theme/DocCardList";

# Importing model predictions

:::caution

When running an importer, any previously imported predictions will be overwritten!
To ensure the ability to revert to previous model iterations, it is important to [version your projects][project-versioning]. 

:::

If you aren't familiar with how to build lists of `Prediction` objects, please have a look at the [Import model predictions][import-predictions-guide] guide first.
It will show you how to construct predictions for bounding boxes, polygons, masks, and classifications.

With these predictions in hand, importing them is done as follows:

```python
from pathlib import Path

from encord_active.lib.db.predictions import Prediction
from encord_active.lib.model_predictions.importers import import_predictions
from encord_active.lib.project import Project

project_path = Path("/path/to/your/project/root")

predictions: list[Prediction] = ...  # Your list of predictions

import_predictions(Project(project_path), predictions)
```

## Other Options

There are a couple of additional ways to import predictions that are stored in common file formats.
You can find them here:

<DocCardList />


[import-predictions-guide]: ../../import/import-predictions
[project-versioning]: ../../user-guide/versioning
