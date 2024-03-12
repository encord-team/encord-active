---
title: "Versioning"
slug: "active-versioning"
hidden: true
metadata: 
  title: "Versioning"
  description: "Learn data versioning in Encord Active: Track experiments, models, and project states. Global checkpoints for effective comparisons"
  image: 
    0: "https://files.readme.io/e269354-image_16.png"
createdAt: "2023-07-12T09:21:53.085Z"
updatedAt: "2023-08-11T13:54:16.572Z"
category: "6480a3981ed49107a7c6be36"
---

**Learn how to version your data, labels, and models in Encord Active**

### Why do you version your data?
When you do experiments and test hypotheses, you typically want to jump back and forth between different versions of your data, labels, and models. For example, when you train a model on a specific subset of your data, you will typically find that there is an edge case for which your model performs poorly. Hence, you expand your dataset with more data to better cover the edge case and train a new model.

In order to track your experiments and compare not only the model performance but also the underlying data shown to the model, you can use the project versioning feature of Encord Active.

### What types of versioning are supported?

The versioning is global for the project, covering everything from the available data and labels at a specific moment to the corresponding model predictions. This ensures that all relevant information is versioned and accessible.

Currently, versioning operates through checkpoints, allowing you to create checkpoints and navigate between them to review previous states of the project.


### How do I version my data?

#### Creating a new version


In order to create a new version, navigate to the toolbox in the explorer pages, access the _Version_ tab, provide a version name and click the <kbd>Create</kbd> button.

![Version creation form](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/version-creation-form.png)

> 👍 Tip
> You also have the ability to discard any outstanding changes, i.e. everything after the last version.


#### Viewing a previous version

On the left sidebar, there is a drop-down which allows version selection. Selecting an old version will temporarily save any outstanding changes until the latest version is selected again.

> 🚧 Caution
> While on a previous version the app will be in read-only mode. Any changes made will be discarded.