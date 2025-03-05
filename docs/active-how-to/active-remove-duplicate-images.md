---
title: "Remove duplicate images"
slug: "active-remove-duplicate-images"
hidden: false
metadata: 
  title: "Removing duplicate images"
  description: "Enhance dataset quality: Detect and remove duplicate images with Encord Active. Mitigate bias and optimize data for models."
  image: 
    0: "https://files.readme.io/05b71a8-image_16.png"
createdAt: "2023-07-11T16:27:42.223Z"
updatedAt: "2023-08-09T16:11:47.793Z"
category: "6480a3981ed49107a7c6be36"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/hosted_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

The presence of duplicate or closely similar images can introduce bias in deep learning models. Encord Active provides the capability to identify and eliminate duplicate or near-duplicate images from datasets. This process contributes to enhancing data quality by removing redundant instances, ultimately leading to improved model performance.

In this workflow, the [`Uniqueness` quality metric](https://docs.encord.com/docs/active-data-quality-metrics#uniqueness) is used to identify duplicate and near-duplicate images.

## Uniqueness metric

The `Uniqueness` metric evaluates all images within the dataset and assigns a uniqueness score to each, indicating their distinctiveness.

- The uniqueness score falls within the [0,1] range. A higher score indicates a greater level of image uniqueness. The **Duplicates** summary on the **Data > Overview** tab uses a range between 0 and 0.0001.

- A score of zero signifies the presence of at least one identical image within the dataset. For instances with _N_ duplicate images, _N-1_ of them are assigned a score of zero (with only one holding a non-zero score) to facilitate their exclusion from the dataset.

- Near-duplicate images are labeled as `Near-duplicate image` and are presented side by side in the Explorer's grid view. This setup simplifies the decision-making process when selecting which image to keep and which one to remove.

<!---
[//]: # (TODO add this option when Description fields made by metrics are accessible in the UI)
[//]: # (- In the context of _N_ duplicate images, _N-1_ images are associated with a single representative image, visible through the `Description` field in the details of each image.)
--->

## Quick Tour

All the sections in the Quick Tour assume that you are already in a Project.

> 👍 Tip
> Choose any image in the Explorer workspace and click its _Similar items_ [!Similarity button](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/similarity-button.png) button. This displays images similar to the selected one, including any duplicates if they exist.

### Explorer

The _Explorer_ page has three areas that can help you find duplicate images in your Project.

<details>

<summary><b>1: Duplicates Shortcut</b></summary>

Found in the _Overview_ tab, any images that have a `Uniqueness` value of 0 to 0.0001 are highlighted as duplicates. You can adjust this value from the _Filter_ tab.

![Duplicates shortcut](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/image-duplicates-qt-01.png)

</details>

<details>

<summary><b>2: Sorting by `Uniqueness`</b></summary>

The entire Project can be sorted by `Uniqueness`. Sort by ascending order to display duplicates first.

![Sorting by `Uniqueness`](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/image-duplicates-qt-02.png)

</details>

<details>

<summary><b>3: Filtering by `Uniqueness`</b></summary>

Filter the entire project using `Uniqueness`. 

Go to **Filter** tab > **Add Filter** > **Data Quality Metrics** > **Uniqueness** A small histogram diagram appears above the filter.

You can then change the filter settings to specify a range closer to 0.

![Filtering by `Uniqueness`](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/image-duplicates-qt-03.png)

</details>

### Analytics

In a Project, go to the _Analytics_ page and pick the `Uniqueness` quality metric for the _Metric Distribution_ section.

![Distribution of data based on Uniqueness scores]([!Duplicates shortcut](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/image-duplicates-qt-anal-01.png))

The chart displays the distribution of data based on the `Uniqueness` scores.

## Remove duplicate images

When you want to remove/exclude duplicate images from a dataset, tag duplicate images and create a Collection devoid of duplicates.

<details>

<summary><b>To remove duplicate images from your Project:</b></summary>

1. Log in to the Encord platform.
   The landing page for the Encord platform appears.
   
2. Click **Active** in the main menu.
   The landing page for Active appears.

3. Click the Project.
   The landing page for the Project appears with the _Explorer_ tab selected with _Data_ selected.

4. Click the _Duplicates_ shortcut under the _Overview_ tab.
   The _Duplicates_ shortcut applies the `Uniqueness` filter to all images in the Project. The `Uniqueness` filter returns images with a `Uniqueness` value between 0 and 0.0001.

5. Sort the filtered data in ascending order by `Uniqueness`.

6. Adjust the `Uniqueness` filter from the default value to find all the duplicate images in the Project.
   As you adjust the filter the images that appear in the Explorer workspace change.

7. Select one and then all images.

8. Click the **Add to a Collection** button to create a Collection.

9. Click **New Collection**. 

10. Name the Collection `Duplicates`.
    All selected images have the tag `Duplicates` applied to them.

11. Reset all Filters.

12. Add a Collections filter that excludes `Duplicates`.

7. Select one and then all images.

8. Click the **Add to a Collection** button to create a Collection.

9. Click **New Collection**. 

10. Specify a meaningful name for the Collection.

11. Go to the _Collections_ page.

12. Select the Collection that excludes `Duplicates`. 

13. Click **Create Dataset**.

14. Specify a meaningful name and description for the Dataset and Project.

15. Click **Submit**.
    The Dataset and Project appear in Annotate.

Incorporating this workflow into dataset management strategies can significantly enhance data quality, eliminate redundancies, and contribute to more accurate model training and evaluation.

</details>

## Remove near-duplicate images

[block:image]
{
  "images": [
    {
      "image": ["https://storage.googleapis.com/docs-media.encord.com/static/img/images/workflows/improve-your-data-and-labels/remove-duplicate-images/remove-duplicate-images-11.png",
      "",
      "An example of near-duplicate image pairs detected with Encord Active"
      ],
      "align": "center",
      "caption": "An example of near-duplicate image pairs detected with Encord Active"
    }
  ]
}
[/block]

Similar to duplicates, near-duplicate images are images where one image slightly differs from another due to shifts, blurriness, or distortion. Consequently, they should also be eliminated from the dataset. However, in this scenario, a decision is required to determine which sample remains and which is discarded. These images possess scores marginally greater than 0 and are displayed alongside one another in the Explorer grid view workspace, facilitating easy comparison.

1. Log in to the Encord platform.
   The landing page for the Encord platform appears.
   
2. Click **Active** in the main menu.
   The landing page for Active appears.

3. Click the Project.
   The landing page for the Project appears with the _Explorer_ tab selected with _Data_ selected.

4. Click the _Duplicates_ shortcut under the _Overview_ tab.
   The _Duplicates_ shortcut applies the `Uniqueness` filter to all images in the Project. The `Uniqueness` filter returns images with a `Uniqueness` value between 0 and 0.0001.

5. Sort the filtered data in ascending order by `Uniqueness`.

6. Adjust the `Uniqueness` filter from the default value to **0 to 0.05**.

7. Examine the images in the Explorer workspace and select the images you want removed from the Project. 

8. Click the **Add to a Collection** button to create a Collection.

9. Click **New Collection**.

   > ℹ️ Note
   > If you already have a Collection called `Duplicates`, add the images to the existing Collection and go to _step 11_.

10. Name the Collection `Duplicates`.
    All selected images have the tag `Duplicates` applied to them.

11. Reset all Filters.

12. Add a Collections filter that excludes `Duplicates`.

13. Select one and then all images.

14. Click the **Add to a Collection** button to create a Collection.

15. Click **New Collection**. 

16. Specify a meaningful name for the Collection.

17. Go to the _Collections_ page.

18. Select the Collection that excludes `Duplicates`. 

19. Click **Create Dataset**.

20. Specify a meaningful name and description for the Dataset and Project.

21. Click **Submit**.
    The Dataset and Project appear in Annotate.

With these actions, users can efficiently manage near-duplicate images and improve dataset quality.
