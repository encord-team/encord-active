---
title: "Import Predictions"
slug: "active-import-model-predictions-cloud"
hidden: false
metadata: 
  title: "Import Model Predictions to Active Cloud"
  description: "Assess model quality with Encord Active Cloud analytics and metrics. Optimize model evaluation."
category: "6480a3981ed49107a7c6be36"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/hosted_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

---

Encord Active does not only provide a streamlined method to currate your image data, Active also provides metrics and analytics to optimize your model's performance. Simply upload your model's predictions in to Active Cloud to start.

Your predictions must be imported to Active, before you can use the Predictions feature on the _Explorer_ page and the _Model Evaluation_ page.

## STEP 1: Prepare Your Predictions for Import

> :warning: **Disclaimer**: We strongly recommend that you are knowledgeable about the `Encord` SDK. If you are unfamiliar with the SDK or if you do not understand the following boilerplate code, refer to this topic in the [SDK documentation](https://docs.encord.com/reference/sdk-import-labels-annotations).

Within Encord Active, predictions use the same format as labels. For the most part, creating predictions programmatically is the same as creating labels. 

We'll start from the "Label Creation Boilerpate", which shows you how to upload labels into Encord. Then we'll show you how to modify the boilerplate to store predictions. 

You can create labels programmatically in Encord following this structure:

```python Label Creation Boilerplate

# Import dependencies
import os

from encord import EncordUserClient, Project
from encord.objects import LabelRowV2

# Authenticate client and identify project
ssh_private_key_path = os.getenv("ENCORD_CLIENT_SSH_PATH")
project_hash = os.getenv("ENCORD_PROJECT_HASH")

assert ssh_private_key_path is not None
assert project_hash is not None

client = EncordUserClient.create_with_ssh_private_key(ssh_private_key_path)
project: Project = client.get_project(project_hash)

# Add labels
def add_information_to_lr(lr: LabelRowV2):
    ...  # Logic for adding labels/predictions

# Save labels
label_rows: list[LabelRowV2] = project.list_label_rows_v2()
for lr in label_rows:
    lr.initialise_labels()
    add_information_to_lr(lr)
    lr.save()
```

> ℹ️ Note
> 
>The documentation to add labels to the label row is available [here](https://docs.encord.com/reference/sdk-working-with-labels).

To store the labels as predictions instead, you need to change the following things in the Label Creation Boilerplate:

-  Initialize the label rows without the existing labels [see Storing Predictions Boilerplate line 27-30]
- Store the predictions as serialized json [see Storing Predictions Boilerplate line 32-36]
- Make `add_information_to_lr` use your model to "create labels" (remember that labels and predictions are equivalent in terms of structure) [see Storing Predictions Boilerplate line 19]

```python Store Predictions Boilerplate
# Import dependencies
import os

from encord import EncordUserClient, Project
from encord.objects import LabelRowV2

# Authenticate client and identify project
ssh_private_key_path = os.getenv("ENCORD_CLIENT_SSH_PATH")
project_hash = os.getenv("ENCORD_PROJECT_HASH")

assert ssh_private_key_path is not None
assert project_hash is not None

client = EncordUserClient.create_with_ssh_private_key(ssh_private_key_path)
project: Project = client.get_project(project_hash)

# Make `add_information_to_lr` use your model to "create labels"
def add_information_to_lr(lr: LabelRowV2):
    ...  # Logic for adding labels/predictions


label_rows: list[LabelRowV2] = project.list_label_rows_v2()
serialized_output: list[dict] = []
for lr in label_rows:

# Initialize the label rows without the existing labels
    lr.initialise_labels(  # ignore existing labels
        include_object_feature_hashes=set(),
        include_classification_feature_hashes=set(),
    )
# Store the predictions as serialized json
    serialized_output.append(lr.to_encord_dict())  # Serialize

import json
with open("predictions.json", "w") as f:
    json.dump(serialized_output, f)

```

Now that you have the `predictions.json` file, you can move to STEP 2 and import the JSON file into the Active UI.

## STEP 2: Import Predictions Set

Once you have the `predictions.json` file from STEP 1, Prediction Sets can be imported from both the _Model Evaluation_ page and the **Upload predictions** button ( **+** ) on the _Overview_ tab of the _Predictions_ page in the _Explorer_ page.

<details>

<summary><b>To import Prediction Sets into Active from the Model Evaluation page:</b></summary>

1. Contact Encord to get started with Encord Active.

2. Log in to the Encord platform.
   The landing page for the Encord platform appears.

3. Click **Active** in the main menu.
   The landing page for Active appears.

4. Click the Project.
   The landing page for the Project appears with the _Explorer_ tab selected.

5. Click the **Model Evaluation** tab.
   The _Model Evaluation page_ appears.

   ![Import Model Predictions](https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/import-predictions.png)

6. Click the **Import prediction** button.
   The _Upload predictions_ dialog appears.

7. Type a meaningful name for the prediction.

8. Click the **Select Predictions File** button.
   A dialog box appears.

9. Select the JSON file to upload.

10. Click **Open**.

11. Click **Start Upload**.
    Once the upload completes the _Model Evaluation_ page and the _Predictions_ page on the _Explorer_ page are available for use.

</details>

> ℹ️ Note
> 
> If you have any issues importing your predictions contact your CSM or contact us at support@encord.com.

## Next Steps

### Model and Prediction Validation

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Clickable Div</title>\n    <style>\n        .clickable-div {\n            display: inline-block;\n            width: 200px;\n            height: 50px;\n            background-color: #ffffff;\n            border: solid;\n            text-align: center;\n            line-height: 50px;\n            color: #000000;\n            text-decoration: none;\n            margin: 10px;\n        }\n\n        .clickable-div:hover {\n            background-color: #ededff;\n        }\n    </style>\n</head>\n<body>\n    <a href=\"https://docs.encord.com/docs/active-import-from-annotate\" class=\"clickable-div\">1. Import from Annotate</a> <a href=\"https://docs.encord.com/docs/active-model-predictions-eval\" class=\"clickable-div\">3. Review Prediction Metrics</a> <a href=\"https://docs.encord.com/docs/active-create-collections\" class=\"clickable-div\">4. Create Collection</a> <a href=\"https://docs.encord.com/docs/active-send-collection-to-annotate\" class=\"clickable-div\">5. Send to Annotate</a> <a href=\"https://docs.encord.com/docs/active-sync-with-annotate\" class=\"clickable-div\">6. Sync with Annotate</a> <a href=\"https://docs.encord.com/docs/active-update-collections\" class=\"clickable-div\">7. Update Collection</a>\n</body>\n</html>"
}
[/block]