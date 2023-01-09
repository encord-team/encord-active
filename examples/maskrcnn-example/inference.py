import matplotlib.pyplot as plt
import torch
from PIL import Image
from utils.encord_dataset import EncordMaskRCNNDataset
from utils.model_libs import get_model_instance_segmentation
from utils.provider import get_config, get_transform

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

params = get_config("config.ini")

dataset_validation = EncordMaskRCNNDataset(
    img_folder=params.inference.target_data_folder,
    ann_file=params.inference.target_ann,
    transforms=get_transform(train=False),
)

model = get_model_instance_segmentation(len(dataset_validation.coco.cats) + 1)
model.load_state_dict(torch.load(params.inference.model_checkpoint_path))
model.to(device)
img, target, img_metadata = dataset_validation[4]

model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

org_img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
pred_mask = Image.fromarray(prediction[0]["masks"][0, 0].mul(255).byte().cpu().numpy())

plt.imshow(org_img)
plt.tight_layout()
plt.show()

plt.imshow(pred_mask)
plt.tight_layout()
plt.show()
