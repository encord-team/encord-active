---
sidebar_position: 2
---

# Find Similar Images

**Mine edge cases, duplicates, and check the quality of your labels with visual similarity search**

Often, when you find characteristics of interest in your dataset you may want to identify other similar images (e.g., to send for relabelling or deletion).
Identifying these cases can help you understand where your data is, e.g., underrepresented or mislabelled. 
As your dataset grows, finding such cases manually becomes increasingly difficult. Using Encord Active's **similarity search** you can easily find semantically similar images in your dataset. 
When have identified an edge case or a duplicate, you can tag it and export or delete it in the _Actions tab_.

 `Prerequisites:` Dataset & Embeddings 

:::tip

 There are two tabs where you can utilize similarity search: [Summary tab](/pages/data-quality/summary) and [Explorer tab](/pages/data-quality/explorer).

:::

### Steps:
1. Navigate to the _Data Quality_ > _Summary_ or _Explorer_ tab.
2. Select a metric in the top left menu to order your data by.
3. Select an image of interest and click _Show Similar Images_. 
   * Encord Active shows the most semantically similar images below the image in a new tab.
4. If you have not yet created any tags, create a tag on the left sidebar.
5. Tag images of interest.
6. Repeat step 2-5 until satisfied.
7. When you are satisfied with your selection navigate to the _Actions_ tab to act.
8. Within the _Actions_ tab Click _Add filters_, scroll the bottom, and select _tags_. Next, choose the tags you would like to export, relabel, delete, or augment from your dataset.
   * To export images in CSV click _Download filtered data_.
   * To delete images from your dataset click _Delete_.
   * To relabel images please contact Encord to hear more.
   * To augment similar images please contact Encord to hear more.
