import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.as_posix())
import torch
import torchvision
from provider import convert_coco_poly_to_mask
import logging


class EncordMaskRCNNDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img_metadata = self.coco.loadImgs(self.ids[idx])

        image_id = self.ids[idx]
        img_width, img_height = img.size

        boxes, labels, area, iscrowd = [], [], [], []
        for target_item in target:
            if target_item["bbox"][2] <= 0 or target_item["bbox"][3] <= 0:
                logging.warning(
                    f"ERROR: Target bbox for the following image \n"
                    f'title:      {img_metadata[0]["image_title"]} \n'
                    f'label_hash: {img_metadata[0]["label_hash"]} \n'
                    f'data_hash:  {img_metadata[0]["data_hash"]} \n'
                    f'has non-positive width/height => [x,y,w,h]: {target_item["bbox"]}. \n'
                    f"Therefore, skipping this annotation."
                )
                continue
            boxes.append(
                [
                    target_item["bbox"][0],
                    target_item["bbox"][1],
                    target_item["bbox"][0] + target_item["bbox"][2],
                    target_item["bbox"][1] + target_item["bbox"][3],
                ]
            )

            labels.append(target_item["category_id"])
            area.append(target_item["bbox"][2] * target_item["bbox"][3])
            iscrowd.append(target_item["iscrowd"])

        segmentations = [obj["segmentation"] for obj in target]
        masks = convert_coco_poly_to_mask(segmentations, img_height, img_width)

        processed_target = {}
        processed_target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        processed_target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        processed_target["masks"] = masks
        processed_target["image_id"] = torch.tensor([image_id])
        processed_target["area"] = torch.tensor(area)
        processed_target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self._transforms is not None:
            img, processed_target = self._transforms(img, processed_target)

        return img, processed_target, img_metadata

    def __len__(self):
        return len(self.ids)
