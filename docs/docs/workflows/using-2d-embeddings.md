---
sidebar_position: 3
---

# Using 2D Embeddings

**Use 2D embeddings to leverage your active learning cycle**

Encord Active presents you 2D embedding plot for all your images in your project. You can use it to find interesting 
clusters and understand your data better, weakly label your images, remove detrimental images and many more. 2D 
embedding plot can be viusalized in the **Explorer** pages of Data Quality and Label Quality tabs. Here is an example
embedding plot for the COCO 2017 validation set.

![using-2d-embeddings-1](../images/workflows/using-2d-embeddings/using-2d-embeddings-1.png)

You see how images can be clustered around certain regions. In order to visualize images in these clusters and learn
what is common among these samples you can use Box or Lasso Select on the upper right corner of the plot.

![using-2d-embeddings-2](../images/workflows/using-2d-embeddings/using-2d-embeddings-2.gif)

Once you selected a region the rest of the app (data distribution plot and main grid view of images) will be updated
accordingly. Now you can only visualize the images in the selected region. For the above selection, selected region has
snowy images.

![img.png](../images/workflows/using-2d-embeddings/using-2d-embeddings-3.png)

Now you can perform several actions with the selected group:
- You can tag them and export in the **Actions** => **Filter & Export** page so that you can label these images 
automatically via a script.
- If you do not want them in your dataset for some reason, you can tag the all images and remove the tags from the
selected ones so that you can create a new version of your dataset without them. You can specifically apply this step
for the images that you think including them will be detrimental.