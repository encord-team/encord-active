---
title: "In Encord"
slug: "active-in-encord"
hidden: true
metadata: 
  title: "In Encord"
  description: "Set up active learning process in Encord. Streamline workflow stages: Initialization, high-value data prioritization, model training. Optimize learning."
  image: 
    0: "https://files.readme.io/657055d-image_16.png"
createdAt: "2023-07-11T16:27:41.851Z"
updatedAt: "2023-08-09T13:52:08.839Z"
category: "6480a3981ed49107a7c6be36"
---
**Learn how to set up the components of your active learning process in Encord**

> ℹ️ Note
> Active learning workflows in the Encord platform are specifically designed for [workflow projects](https://docs.encord.com/docs/annotate-annotation-projects). This requirement allows for seamless task movement between essential stages such as `label`, `review` and `complete` when utilizing the SDK.


Active learning workflows in Encord Active share the following key stages:

1. [Initialization](#initialization).
2. [Prioritizing high-value data to label](#prioritize-high-value-data-to-label).
3. [Model training and update](#model-training-and-update).

If you prefer to witness an active learning workflow in action, please take a look at the end-to-end tutorial for [MNIST](https://docs.encord.com/docs/active-easy-active-learning-mnist).

## Initialization
To start an active learning workflow, you need an initial labeled dataset for training the machine learning model. In the Encord platform, this corresponds to having a project with annotations.

If you don't have any projects yet, you can watch the tutorial video on setting up a [workflow project](https://docs.encord.com/docs/annotate-workflows-and-templates) to get started quickly.

### Choose an Encord project

To proceed, you should pull the project into Encord Active. Execute the following CLI command and remember to acknowledge that you would like to include uninitialized label rows, as they represent unannotated data.

```bash
encord-active import project
```

If you require detailed information on the options available during the import process, you can refer to the [Import from Encord platform](https://docs.encord.com/docs/active-import-encord-project) guide.  

If your workflow project already contains annotations, you can proceed directly to [Model training and update](#model-training-and-update).

[//]: # (Alternatively, if you have a trained model that you intend to use in the active learning workflow, you can jump to the [Querying]&#40;#querying&#41; step.)

## Prioritize high value data to label

If your project does not have any annotations or you are seeking the most appropriate data for labeling, it's essential to score and rank your data.
While random selection is a possibility, Encord Active provides metrics such as [`Image Diversity`](https://docs.encord.com/docs/active-data-quality-metrics#image-diversity) to enhance and optimize annotation impact.
This metric ranks images based on their ease of annotation, enabling prioritization of suitable and manageable data.


> 👍 Tip
> Check out the [quality metrics page](https://docs.encord.com/docs/active-quality-metrics) for a comprehensive overview of available metrics in Encord Active, including the [acquisition functions](https://docs.encord.com/docs/active-acquisition-functions) used for sample selection.


For example, you can follow these steps to prioritize labeling for data with the lowest `Image Diversity` score using the UI:
1. In the _Data Quality_ explorer page, navigate to the toolbox and click on the _Filter_ tab.
2. Select the option that correspond to the first labeling stage (usually named `Annotate 1`) under the `Workflow Stage` metadata filter to pick the unannotated data.
3. Add the `Image Diversity` filter and adjust the slider to select a subset of data with the lowest score.
4. Access the _Action_ tab in the toolbox.
5. Click on the <kbd>🖋 Relabel</kbd> button and follow the instructions to [prioritize labeling](https://docs.encord.com/docs/active-relabeling) for the selected data.


[//]: # (todo remove this info section once Encord Annotate provides task prioritization to the public)
> ℹ️ Note
> Task prioritization for labeling is currently a closed-beta feature in the Encord platform. To learn more about this feature, please reach out to us on [Slack][slack-join] or via [email](mailto:active@encord.com).
> 
> Nevertheless, to mimic the behavior of task prioritization in projects with only single images, you can follow these steps:
> 1. In the _Filter_ tab, select the option that correspond to the first labeling stage (usually named `Annotate 1`) under the `Workflow Stage` metadata filter to pick the data ready to be labeled.
> 2. Use the [bulk tagging](https://docs.encord.com/docs/active-tagging#bulk-tagging) feature to mark them with a data tag, such as `unlabeled`.
> 3. Add the `Image Diversity` filter and adjust the slider to select a subset of filtered data with the lowest score.
> 4. Use the bulk tagging feature to mark this further selection with a data tag, such as `to label next`.
> 5. Reset the filters and choose the `unlabeled` tag option under `Tags`.
> 6. Access the _Action tab_ in the toolbox and click on the <kbd>✅ Mark as Complete</kbd> button and follow the instructions to temporarily move all the selected data to the workflow's `Complete` stage.
> 7. Return to the _Filter_ tab, reset the filters and choose the `to label next` tag option under `Tags`.
> 8. Access the _Action_ tab in the toolbox again, click on the <kbd>🖋 Relabel</kbd> button and follow the instructions to move the selected data to the workflow's first annotation stage.
> 9. Once the selected data has been labeled, use the following filter combination to bring back the remaining data from the `Complete` stage to the first labeling stage as in step (8):
> Select the  `No class` option under `Object Class` and choose the proper tag name (e.g. `unlabeled`) option under `Tags`.
> 
> By following these steps, you can ensure that the first labeling stage contains only the prioritized data for labeling, and the task states align at the end with the flow that utilizes the task prioritization feature.


## Model training and update

In the active learning workflow, model training plays a crucial role. It involves training a machine learning model using the initial labeled dataset and iteratively updating it with newly labeled data. Encord Active provides support for a wide range of models by allowing you to plug in your own model and interface with it using convenient wrappers.

More information can be found in this [here](https://docs.encord.com/docs/active-easy-active-learning-mnist#model-training).


[slack-join]: https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q
