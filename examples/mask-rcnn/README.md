
# Mask R-CNN

This example implements an end-to-end training and evaluation procedure for Mask R-CNN in PyTorch Lightning using Encord Active.

## Contents

This example will go through the following steps:
1. Finetune a [Mask R-CNN](https://pytorch.org/vision/main/models/mask_rcnn.html) model on a dataset following the COCO format. 
This assumes you already have downloaded a dataset and its respective annotations file in the COCO format. 
The best way to do so is through Encord Active and its export function, as described in [the docs](https://encord-active-docs.web.app/pages/export).  
2. Evaluate the trained model on a validation set and upload the validation predictions to Encord Active.
3. Explore model performance with Encord Active and its [Model Assertions](https://encord-active-docs.web.app/category/model-assertions).
This will allow us to identify where and why our model is failing: we will find out what features of the data make the model generate poor predictions.
4. We will then use Encord Active to improve our dataset and get better validation set result.

The goal of this experiment is to show how Encord Active can be used to improve model performance by curating better dataset. 


## Installation

Create a conda virtual environment by directly running the following command in the mask-rcnn folder:

```shell
conda env create -f environment.yml
```

Verify that the new environment is installed correctly:

```shell
conda env list
```

You should see the `maskrcnn` example in the list. Simply activate it with the following command:

```shell
conda activate maskrcnn
```

## Usage

1. Create a config.ini file following the config-example.ini in the root folder. You can use `resize_images` or 
`resize_coco_annotations` in `dataset/dataset_utils.py` to downscale images and corresponding annotations.
2. Run the following command inside the virtual environment:
```shell
python train.py
```
3. (Optional) You can track the model performance on wandb platform.
4. Import predictions to Encord Active platform using the following command:
```shell
encord-active import predictions /path/to/predictions.pkl
```
5. Start Encord Active app and examine the *Model Quality* tab:
```shell
encord-active visualise
```