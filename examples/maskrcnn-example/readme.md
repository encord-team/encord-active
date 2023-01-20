# Encord-Active MaskRCNN Example

This example implements an end-to-end training and evaluation procedure for Mask R-CNN in PyTorch
using Encord Active. 
- The target project should follow the COCO format. You can 
use Encord-Active's export functionality to create COCO annotations.
- Trained model can be evaluated on any dataset using the Encord-Active.

## Installation
Create a new conda virtual environment using the following command
```shell
# inside the project directory
conda env create -f environment.yml
```

Verify that the new environment is installed correctly:
```shell
conda env list
```

You should see the `encord-maskrcnn` example in the list. Simply activate it with the following command:

```shell
conda activate encord-maskrcnn
```

## Training
1. You can use Encord-Active's Actions tab to create COCO annotations.
2. Create a config.ini file by looking at the example_config.ini
3. You can resize the images and corresponding annotations via utils/downscale_dataset.py.
4. For the training, the only required fields are [DATA], [LOGGING], and [TRAIN] sections
5. Activate the environment and run `python train.py`
6. You can track the progress of the training on wandb platform.


## Importing Encord-Active predictions
1. Get the wandb ID from the experiment that you want to use for inference
2. Fill the [INFERENCE] section of the config.ini file
3. Run `python generate_ea_predictions.py` to generate a pickle file.
4. Run the following command to convert pickle file into Encord Active predictions.

```shell
encord-active import predictions /path/to/predictions.pkl -t /path/to/project
```

5. Open the app. Now you can see the model performance on the __Model Quality__ tab.