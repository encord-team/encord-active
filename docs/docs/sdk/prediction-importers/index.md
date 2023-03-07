import DocCardList from "@theme/DocCardList";
import TOCInline from "@theme/TOCInline";
import Tabs from "@theme/Tabs";
import TabItem from "@theme/TabItem";

# Importing Model Predictions

This page shows you how to import model predictions **with code**.

:::caution

Every time you run any of these importers, previously imported predictions will be overwritten!
Make sure to [version your projects][project-versioning] if you want to be able to go back to previous model iterations.

:::

If you aren't familiar with how to build lists of `Prediction` objects, please have a look at [this](../../import/import-predictions) workflow tutorial first.
It will show you how to construct predictions for bounding boxes, polygons, masks, and classifications.

This page will show you how to

1. Run your model over your dataset to get predictions
2. Importing the predictions into Encord Active

## Iterating Over Project Data

There is also a workflow description on importing model predictions [here]

When you have these things in place, there are a couple of options for importing your predictions into Encord Active:

<TOCInline toc={toc.filter((node, index) => node.level === 2 && index <= 10)} />

## Prepare a `.pkl` File to be Imported with the CLI

You can prepare a pickle file (`.pkl`) to be imported with the Encord Active CLI as well.
You do this by building a list of `Prediction` objects.
We support predictions for classifications and object detections.

All predictions need a unique identifier of the data unit (the `data_hash` and potentially a `frame`) and model `confidence` score

In addition to the above, a **classification** prediction contains 3 identifiers, `classification_hash` `attribute_hash` and `option_hash` while an **object detection** prediction contains the `feature_hash`, the actual prediction `data`, and the `format` of that data.

#### Creating a `Prediction` Object

Below, you find examples of how to create an object of each of the three supported types.

<Tabs groupId="data-type">
  <TabItem value="classification" label="Classification" default>

Since we can have multiple classifications for for the same data unit (e.g. `has a dog?` and `has a cat?`) we need to uniquely identify them by providing the 3 hashes from the ontology.

```python
prediction = Prediction(
    data_hash="<your_data_hash>",
    frame = 3,  # optional frame for videos
    confidence = 0.8,
    classification=FrameClassification(
        feature_hash="<your_feature_hash>",
        attribute_hash="<your_attribute_hash>",
        option_hash="<your_option_hash>",
    ),
)
```

:::tip
To find the three hashes, we can inspect the ontology by running

```shell
encord-active print ontology
```

:::

  </TabItem>
  <TabItem value="bbox" label="Bounding Box" default>

You should specify your `BoundingBox` with relative coordinates and dimensions.
That is:

- `x`: x-coordinate of the top-left corner of the box divided by the image width
- `y`: y-coordinate of the top-left corner of the box divided by the image height
- `w`: box pixel width / image width
- `h`: box pixel height / image height

```python
from encord_active.lib.db.predictions import BoundingBox, Prediction, Format, ObjectDetection

prediction = Prediction(
    data_hash="<your_data_hash>",
    frame = 3,  # optional frame for videos
    confidence = 0.8,
    object=ObjectDetection(
        feature_hash="<the_class_id>",
        # highlight-start
        format = Format.BOUNDING_BOX,
        # Your bounding box coordinates in relative terms (% of image width/height).
        data = BoundingBox(x=0.2, y=0.3, w=0.1, h=0.4),
        # highlight-end
    ),
)
```

:::tip

If you don't have your bounding box represented in relative terms, you can convert it from pixel values like this:

```python
img_h, img_w = 720, 1280  # the image size in pixels
BoundingBox(x=10/img_w, y=25/img_h, w=200/img_w, h=150/img_h)
```

:::

  </TabItem>
  <TabItem value="mask" label="Segmentation Mask">

You specify masks as binary `numpy` arrays of size [height, width] and with `dtype` `np.uint8`.

```python
from encord_active.lib.db.predictions import Prediction, Format

prediction = Prediction(
    data_hash =  "<your_data_hash>",
    frame = 3,  # optional frame for videos
    confidence = 0.8,
    object=ObjectDetection(
        feature_hash="<the_class_id>",
        # highlight-start
        format = Format.MASK,
        # _binary_ np.ndarray of shape [h, w] and dtype np.uint8
        data = mask
        # highlight-end
    ),
)
```

  </TabItem>
  <TabItem value="polygon" label="Polygon">

You should specify your `Polygon` with relative coordinates as a numpy array of shape `[num_points, 2]`.
That is, an array of relative (`x`, `y`) coordinates:

- `x`: relative x-coordinate of each point of the polygon (pixel coordinate / image width)
- `y`: relative y-coordinate of each point of the polygon (pixel coordinate / image height)

```python
from encord_active.lib.db.predictions import Prediction, Format
import numpy as np

polygon = np.array([
    # x    y
    [0.2, 0.1],
    [0.2, 0.4],
    [0.3, 0.4],
    [0.3, 0.1],
])

prediction = Prediction(
    data_hash =  "<your_data_hash>",
    frame = 3,  # optional frame for videos
    confidence = 0.8,
    object=ObjectDetection(
        feature_hash="<the_class_id>",
        # highlight-start
        format = Format.POLYGON,
        # np.ndarray of shape [n, 2] and dtype float in range [0,1]
        data = polygon
        # highlight-end
    ),
)
```

:::tip

If you have your polygon represented in absolute terms of pixel locations, you can convert it to relative terms like this:

```python
img_h, img_w = 720, 1280  # the image size in pixels
polygon = polygon / np.array([[img_w, img_h]])
```

Notice the double braces `[[img_w, img_h]]` to get an array of shape `[1, 2]`.

:::

  </TabItem>
</Tabs>

#### Preparing the Pickle File

Now you're ready to prepare the file.
You can copy the appropriate snippet based on your prediction format from above and paste it in the code below.
Note the highlighted line, which defines where the `.pkl` file will be stored.

```python showLineNumbers
from encord_active.lib.db.predictions import Prediction, Format

predictions_to_store = []

for prediction in my_predictions:  # Iterate over your predictions
    predictions_to_store.append(
        # PASTE appropriate prediction snippet from above
    )

# highlight-next-line
with open("/path/to/predictions.pkl", "wb") as f:
    pickle.dump(predictions_to_store, f)
```

In the above code snippet, you will have to fetch the `data_hash`, `class_id`, etc. ready from the for loop in line 5.

#### Import Your Predictions via the CLI

To import the predictions into Encord Active, you run the following command inside the project directory:

```shell
encord-active import predictions /path/to/predictions.pkl
```

This will import your predictions into Encord Active and run all the [metrics](/category/quality-metrics) on your predictions.
With the `.pkl` approach, you are done after this step.

## Predictions from Your Prediction Loop

You probably have a prediction loop, which looks similar to this:

```python
def predict(test_loader):
    for imgs, img_ids in test_loader:
        predictions = model(imgs)
```

You can directly import your predictions into Encord Active by the use of an `encord_active.model_predictions.prediction_writer.PredictionWriter`.
The code would change to something similar to this:

```python
# highlight-next-line
from encord_active.lib.model_predictions.writer import PredictionWriter

def predict(test_loader):
    # highlight-next-line
    with PredictionWriter(data_dir, project) as writer:  # project is defined above.
        for imgs, img_ids in test_loader:
            predictions = model(imgs)
            # highlight-start
            for img_id, img_preds in zip(img_ids, predictions)
                for pred in img_preds:
                    writer.add_prediction(
                        data_hash = img_id,
                        class_uid = pred.class_id,
                        confidence_score = pred.confidence,
                        # either bbox
                        bbox = pred.bbox  # dict with x, y, w, h normalized
                        # or segmentation (mask or normalized polygon points)
                        polygon = pred.mask
                        frame = 0  # If video indicate what frame of the video
                    )
            # highlight-end
```

In the code example above, the arguments to `add_prediction`
are:

- `data_hash`: The `data_hash` of the data unit that the prediction belongs to.
- `class_uid`: The `featureNodeHash` of the ontology object corresponding to the class of the prediction.
- `confidence_score`: The model confidence score.
- `bbox`: A bounding box prediction. This should be a dict with the format:

```python
{
    'x': 0.1  # normalized x-coordinate of the top-left corner of the bounding box.
    'y': 0.2  # normalized y-coordinate of the top-left corner of the bounding box.
    'w': 0.3  # normalized width of the bounding box.
    'h': 0.1  # normalized height of the bounding box.
}
```

- `polygon`: A polygon represented either as a list of normalized `[x, y]` points or a mask of size `[h, w]`.
- `frame`: If predictions are associated with a video, then the frame number should be provided.

:::note

Only one bounding box or polygon can be specified in any given call to this function.

:::

## Automatic Importers for Predictions stored on disk

When you already have your predictions stored in common file formats, you have a couple of options to import them quickly.

Below, you can find the available options.
If you feel that common prediction importers are missing, please reach out to us on [slack][slack-invite] or by [email][ea-email].

<DocCardList />

[ea-email]: mailto:active@encord.com
[slack-invite]: https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q
[project-versioning]: ../../user-guide/versioning
