# Improving the labels using Encord-Active

1. Copy the project on the Encord-Annotate platform (we do not want to lose the original labels).

On Encord-Annotate, open the project, go to Settings => Copy project. Select everything (labels, images etc.) when copying the project.

2. Import the new project to Encord-Active.

See documentation: https://encord-active-docs.web.app/cli/import-encord-project 

3. Open the Encord-Active app.

```shell
(encord-active-environment) > encord-active visualise
```

4. Go to **Label Quality => Explore**, and choose **Object annotation quality** metric.

5. Fix labels via Encord Annotate by clicking the editor button.

6. Once the fixing operation is finished, the local project should be updated again because labels have changed 
on the Encord Annotate side. So delete the local project and import it again.

7. Open the App. Go to **Actions => Filter & Export**, click **Generate COCO file** and when activated 
click **Download filtered data** to download COCO file.

8. Fill in **config.ini** files.

9. Train the model

```shell
# inside project root folder
(encord-maskrcnn) > python train.py
```

10. Import predictions.

```shell
(encord-maskrcnn) > python generate_ea_predictions.py
```

```shell
(encord-active-environment) > encord-active import predictions /path/to/pickle/file -t /path/to/project
```

11. Open the app and check the **Model quality** tab for the details.





