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

## Training
1. Create a config.ini file by looking at the example_config.ini
2. For the training, the only required fields are [DATA], [LOGGING], and [TRAIN] sections
3. Activate the environment and run `python train.py`
4. You can track the progress of the training on wandb platformdd


## Importing Encord-Active predictions
1. Get the wandb ID from the experiment that you want to use for inference
2. Fill the [INFERENCE] section of the config.ini file
3. Run `python generate_ea_predictions.py` to generate a pickle file.
4. Run the following command to convert pickle file into Encord Active predictions.

```shell
encord-active import predictions /path/to/predictions.pkl -t /path/to/project
```

5. Now you can see the model performance on the __Model Quality__ tab.