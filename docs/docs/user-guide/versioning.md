---
sidebar_position: 5
---

# Versioning

**Learn how to version your data, labels, and models in Encord Active.**

## Why do you version your data?
When you do experiments and test hypotheses, you typically want to jump back and forth between different versions of your data, labels, and models. For example, when you train a model on a specific subset of your data, you will typically find that there is an edge case for which your model performs poorly. Hence, you expand your dataset with more data to better cover the edge case and train a new model.

In order to track your experiments and compare not only the model performance but also the underlying data shown to the model, you can use the project versioning feature of Encord Active.

## What types of versioning are supported?

The versioning is global for the project, covering everything from the available data and labels at a specific moment to the corresponding model predictions.
This ensures that all relevant information is versioned and accessible.

Currently, versioning operates through checkpoints, allowing you to create checkpoints and navigate between them to review previous states of the project.


## How do I version my data?

#### Creating a new version

[//]: # (todo check missing steps once the versioning is enabled again, either hosted or local)
In order to create a new version, navigate to the toolbox in the explorer pages, access the _Version_ tab, provide a version name and click the <kbd>Create</kbd> button.

![Version creation form](../images/version-creation-form.png)


:::tip

You also have the ability to discard any outstanding changes, i.e. everything after the last version.

:::

#### Viewing a previous version

On the left sidebar, there is a dropdown which allows version selection. Selecting an old version will temporarily save any outstanding changes until the latest version is selected again.

:::warning

While on a previous version the app will be in read-only mode. Any changes made will be discarded.

:::

