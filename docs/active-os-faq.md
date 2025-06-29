---
title: "Frequently asked questions"
slug: "active-os-faq"
hidden: false
metadata: 
  title: "Frequently asked questions"
  description: "FAQs: Get answers to common queries on Encord Active OS. Streamline information access."
category: "65a71bbfea7a3f005192d1a7"
---

## What is the difference between Encord Active Cloud and Encord Active OS?

Active Cloud is tightly integrated with Encord Annotate, with Active Cloud and Annotate being hosted by Encord. Encord Active OS is an open source toolkit that can be installed on a local computer/server.

Encord Active Cloud and Encord Active OS (open source) are active learning solutions that help you find failure modes in your models, and improve your data quality and model performance.

Use Active Cloud and Active OS to visualize your data, evaluate your models, surface model failure modes, find labeling mistakes, prioritize high-value data for relabeling and more!

---

## Are Annotate and Active Projects the same?

The short answer is no. Here are the differences:

- Annotate Projects are made of Datasets (data), Ontologies, and Workflows. Once Annotation Projects get underway, labels and comments also become part of the Annotate Projects.

If you encounter any issues during the installation process, we recommend checking that you have followed the steps outlined in the [installation guide](https://docs.encord.com/docs/active-oss-install) carefully.

- Active Cloud is integrated with the Encord Plaform, which by extension means Active Cloud is also integrated with Annotate. Active Cloud Projects import portions of Annotate Projects. Active Cloud projects import Annotate Project data, Ontologies, labels, and comments but not Workflows. 

- Active OS is not integrated with the Encord Platform, but still contains some portions of Annotate Projects. Active OS Projects consist of Annotate data and labels.

---

## What is a quality metric?

A quality metric is a function that can be applied to your data, labels, and model predictions to assess their quality and rank them accordingly.

Encord Active uses these metrics to analyze and decompose your data, labels, and predictions.

Here is a [blog](https://encord.com/blog/ai-production-gap/) post on how we like to think about quality metrics.

To learn more about [quality metrics, go here](https://docs.encord.com/docs/active-quality-metrics).

Quality metrics are not only limited to those that ship with Encord Active. In fact, the power lies in defining your own quality metrics for indexing your data just right. [Here](https://docs.encord.com/docs/active-write-custom-quality-metrics) is the documentation for writing your own metrics.

---

## How do I write my own quality metrics?

[See our documentation](https://docs.encord.com/docs/active-write-custom-quality-metrics) on writing your own metrics.

---

## How do I use Encord Active for active learning?

Encord Active supports the active learning process by allowing you to

1. Explore your data to select what to label next
2. Employ acquisition functions to automatically select what to label next
3. Find label errors that potentially harm your model performance
4. Sending data to Encord's Annotation module for labeling
5. Automatically decompose your model performance to help you determine where to put your focus for the next model iteration
6. Tag subsets of data to set aside test sets for specific edge cases for which you want to maintain your model performance between each production model

For detailed information on active learning and the role of Encord Active, you can refer to our documentation on [Active Learning within Encord Active](https://docs.encord.com/docs/active-learning).

---

[//]: # (TODO open this question with an answer to a feature that exists)
[//]: # (## How do I upload my data and labels to Encord Annotate?)

[//]: # ()
[//]: # (Uploading your data to Encord Annotate is as simple as clicking the _Export to Encord_ button on the [Filter and Export]&#40;https://docs.encord.com/docs/active-exporting&#41; page. This will create an ontology, a dataset, and a project on Encord Annotate as well as provide you with links to all three.)

[//]: # ()
[//]: # ()
[//]: # (This action requires an ssh-key associated with Encord Active:)

[//]: # ()
[//]: # (1. [Add your public ssh key]&#40;https://docs.encord.com/docs/annotate-public-keys#set-up-public-key-authentication&#41; to Encord Annotate)

[//]: # ()
[//]: # (2. Associate the private key with Encord Active.)

[//]: # ()
[//]: # (  ```shell)

[//]: # (  encord-active config set ssh-key-pash /path/to/private/key)

[//]: # (  ```)

[//]: # ()
[//]: # (---)

---

## How do I import my model predictions?

To import your model predictions into Encord Active OS, perform the following steps:

1. Build a list of `encord_active.lib.db.predictions.Prediction` objects that represent your model predictions.
2. Store the list of predictions in a pickle file.
3. Run the command `encord-active import predictions /path/to/your/file.pkl`, where `/path/to/your/file.pkl` is the path to the pickle file containing your predictions.

By executing this command, Encord Active imports and incorporates your model predictions into the project. You can refer to the [workflow description](https://docs.encord.com/docs/active-import-model-predictions) for importing model predictions for more detailed instructions.

---

## How can I do dataset management with Encord Active?

Dataset management can be done in Active OS the following ways:

- You can [tag](https://docs.encord.com/docs/active-tagging) your data to keep track of subsets (or versions) of your dataset.

- If you are planning to do more involved changes to your dataset and you want the ability to go back, your best option is to use the <kbd>Clone</kbd> button in the _Action_ tab of the toolbox in the application's explorer pages.

---

## How do I version my data and labels through Encord Active?

While you can version your data and labels using Active, Annotate supports [data versioning](https://docs.encord.com/docs/annotate-annotation-projects#save-label-version) and [label versioning](https://docs.encord.com/docs/annotate-annotation-projects#saved-versions).

The best way to version your project in Active OS is to tag your data with the [tags](https://docs.encord.com/docs/active-tagging) feature as you go.

Alternatively, you can use `git`. To do that, we suggest adding a `.gitignore` file with the following content:

```gitignore
data/**/*.jpg
data/**/*.jpeg
data/**/*.png
data/**/*.tiff
data/**/*.mp4
```

After that, run the following: `git add .; git commit -am "Initial commit"`.

---

## How do I use Encord Active to find label errors?

[See this blog post](https://encord.com/blog/find-and-fix-label-errors-tutorial/) on finding and fixing label errors using Encord Active OS.

---

## Can I use Encord Active without an Encord account

Yes, you can use Encord Active OS without an Encord Account. Encord Active OS is an open source project aimed to support all computer vision based active learning projects. For example:
- Use the [`init`](https://docs.encord.com/docs/active-cli#init) command to initialize a project from an image directory.
- [Import a COCO project](https://docs.encord.com/docs/active-import-coco-project).

Please see our [import documentation](https://docs.encord.com/docs/active-import-coco-project) for more details, and options available to you.

---

## Does data stay within my local environment?

**Yes!**

When using Active OS everything you do with the library stays within your local machine. No statistics, data, or other information is collected or sent elsewhere.

The only communication that occurs with the outside world is with Encord's main platform - if you have a project linked to Encord.

---

## What do I do if I have issues with the installation of Active OS?

If you encounter any issues during the Active OS installation process, we recommend checking that you have followed the steps outlined in the [installation guide](https://docs.encord.com/docs/active-oss-install) carefully.

If the problem persists or if you have any further questions, please don't hesitate to get in touch with us via [Slack][slack-join] or [email](mailto:active@encord.com). We'll be happy to assist you with any installation-related issues you may have.

---

## How do I add my own embeddings?

Please see this [notebook](https://colab.research.google.com/github/encord-team/encord-active/blob/main/examples/adding-own-custom-embeddings.ipynb) to learn how to add your own custom embeddings using Active OS.

---

## What is the tagging feature in Active OS?

In the Active OS UI, throughout the Data Quality, <<glossary:Label>> Quality, and Model Quality pages, you can tag your data.
There are two different levels at which you can tag data; the data level which applies to the raw images/video frames and the label level which applies to the <<glossary:classification>>s and objects associated to each image.

You can, for example, use tags to [filter](https://docs.encord.com/docs/active-filtering) your data for further processing like relabeling, training models, or inspecting model performance based on a specific subset of your data.

[Here](https://docs.encord.com/docs/active-tagging) is some more documentation on using the tagging feature.

---

## What should I do if I encounter an error?

If you come across an error, don't worry! We're here to assist you.
Reach out to us on [Slack][slack-join] or shoot us an [email](mailto:active@encord.com), and we'll promptly address your concern.

Additionally, we greatly appreciate it if you could [report the issue](https://github.com/encord-team/encord-active/issues/new/choose) on GitHub. Your feedback and bug reports help us improve Encord Active for everyone.


[ea-lib]: https://github.com/encord-team/encord-active/tree/main/src/encord_active/lib
[ea-server]: https://github.com/encord-team/encord-active/tree/main/src/encord_active/server
[slack-join]: https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q

---

## How does Encord Active OS integrate without the Encord Platform?

There are multiple ways in which you can integrate your data with Encord Active. We have described how to import data [here](https://docs.encord.com/docs/active-import). To integrate model predictions, you can read more [here](https://docs.encord.com/docs/active-import-model-predictions).

Exporting data back into the rest of your pipeline can be done using the toolbox in the application's explorer pages.

---

## Initializing Encord Active OS is taking a long time, what should I do?

For larger projects, initialization of Active OS can take a while.
While we're working on improving the efficiency, there are a couple of tricks that you can do.

1. As soon as the metric computations have started (indicated by Encord Active printing a line containing `Running metric`) you can open a new terminal and run `encord-active start`. This will allow you to continuously see what have been computed so far. Refresh the browser once in a while when new metrics are done computing in your first terminal.

2. You can also kill the import process as soon as the metrics have started to compute. This will leave you with a project containing fewer quality metrics. As a consequence, you will not be able to see as many insights as if the process is allowed to finish. However, you can always use the [`encord-active metric run`](https://docs.encord.com/docs/active-cli#run) command to run metrics that are missing.

---

## Can I use Encord Active OS without a UI?

Yes, you can!

The code base is structured such that all data operations live in [`encord_active.lib`][ea-lib] and [`encord_active.server`][ea-server] which serve as the "backend" for the UI. As such, everything you can do with the UI can also be done by code.

Other good resources can be found in our [example notebooks](https://github.com/encord-team/encord-active/tree/main/examples).

---