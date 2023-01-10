import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.as_posix())

import configparser
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
import torchvision
import transforms as T
from pycocotools import mask as coco_mask


def get_config(path: str):
    """Read config from file.
    :param path: path to config file.
    :return: nested SimpleNamespace object with each section of the config.ini.
    """
    config = configparser.ConfigParser()
    config.read(path)
    params = {}
    for section in config.sections():
        d = {}
        for key in config[section]:
            # Convert each value to the appropriate type
            try:
                value = eval(config[section][key])
            except:
                value = config[section][key]
            if section == "PATHS":
                value = Path(value).expanduser()
            d[key] = value
        params[section.lower()] = SimpleNamespace(**d)
    return SimpleNamespace(**params)


def get_transform(train: bool):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def threshold_masks(outputs: List[Dict[str, torch.Tensor]], threshold: float = 0.5):
    thresh_outs = []
    for out in outputs:
        out["masks"] = (out["masks"] > threshold).squeeze()
        if len(out["masks"].shape) == 2:
            out["masks"] = out["masks"].unsqueeze(0)
        thresh_outs.append(out)
    outputs = thresh_outs
    return outputs


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0] or anno[0]["keypoints"] is None:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    if not isinstance(dataset, torchvision.datasets.CocoDetection):
        raise TypeError(
            f"This function expects dataset of type torchvision.datasets.CocoDetection, instead  got {type(dataset)}"
        )
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def setup_reproducibility(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
