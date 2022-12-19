import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from dataset.coco_dataset import EncordCocoDetection
from dataset.coco_utils import coco_remove_images_without_annotations
from pycocotools import mask as mask_utils
from torch import Tensor
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def ann_to_rle(ann: dict, *, img_height: int, img_width: int):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann["segmentation"]
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_utils.frPyObjects(segm, img_height, img_width)
        rle = mask_utils.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = mask_utils.frPyObjects(segm, img_height, img_width)
    else:
        # rle
        rle = ann["segmentation"]
    return rle


def ann_to_mask(ann: dict, img_height: int, img_width: int):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = ann_to_rle(ann, img_height=img_height, img_width=img_width)
    m = mask_utils.decode(rle)
    return m


class CocoMaskRCNNTransforms(nn.Module):
    """
    Custom transforms class for MaskRCNN

    Args:
        resize_size (Optional[Union[int, list]]): The size of the image after resizing. If int, use the same resize for both height and width. Defaults to None.
        interpolation (InterpolationMode, optional): Interpolation mode. Defaults to InterpolationMode.BILINEAR.

    Returns:
        Tuple[Tensor, List[dict]]: Returns the image and the target with the masks resized and the bounding boxes rescaled.
    """

    def __init__(
        self,
        *,
        resize_size: Optional[Union[int, list]] = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        if isinstance(resize_size, int):
            resize_size = [resize_size, resize_size]
        self.resize_size = resize_size
        self.interpolation = interpolation

    def forward(self, img: Tensor, target: List[dict]) -> Tuple[Tensor, List[dict]]:
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img_height, img_width = img.shape[-2:]

        for obj_dict in target:
            mask = ann_to_mask(obj_dict, img_height=img_height, img_width=img_width)
            if self.resize_size is not None:
                mask = F.resize(
                    torch.Tensor(mask).unsqueeze(0),
                    self.resize_size,
                    interpolation=self.interpolation,
                ).squeeze()

            obj_dict["masks"] = mask

            width_factor = self.resize_size[0] / img_width if self.resize_size is not None else 1.0
            height_factor = self.resize_size[1] / img_height if self.resize_size is not None else 1.0

            obj_dict["bbox"] = [
                obj_dict["bbox"][0] * width_factor,
                obj_dict["bbox"][1] * height_factor,
                (obj_dict["bbox"][0] + obj_dict["bbox"][2]) * width_factor,
                (obj_dict["bbox"][1] + obj_dict["bbox"][3]) * height_factor,
            ]

        if self.resize_size is not None:
            img = F.resize(img, self.resize_size, interpolation=self.interpolation)

        return img, target

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``. "
            f"Finally the values are first rescaled to ``[0.0, 1.0]``."
        )


def maskrcnn_collate_fn(batch):
    """
    During training, the Mask-RCNN expects:
        - images: a list of tensors, each of shape ``[C, H, W]`` one for each image, and in ``0-1`` range.
        - Targets (list of dictionary), containing:
            - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
              ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            - labels (``Int64Tensor[N]``): the class label for each ground-truth box
            - masks (``UInt8Tensor[N, H, W]``): the segmentation binary masks for each instance

    :param batch: list of tuples (image, target) where target is a list of dictionaries
    :return: a tuple of requires list of images and processed targets
    """
    images, targets, metadata = list(zip(*batch))
    processed_targets: List[Dict[str, torch.Tensor]] = []
    for obj_list in targets:
        boxes_list, labels_list, masks_list, image_id_list = [], [], [], []
        for obj_dict in obj_list:
            boxes_list.append(obj_dict["bbox"])
            labels_list.append(obj_dict["category_id"])
            masks_list.append(torch.Tensor(obj_dict["masks"]))
            image_id_list.append(obj_dict["image_id"])
        processed_targets.append(
            {
                "boxes": torch.tensor(boxes_list),
                "labels": torch.tensor(labels_list),
                "masks": torch.stack(masks_list),
                "image_id": image_id_list,
            }
        )
    return images, processed_targets, metadata


def get_dataloader(
    root: Union[str, List[str]],
    annFile: Union[str, List[str]],
    *,
    shuffle: Union[List[bool], bool] = False,
    resize_size: int = 512,
    batch_size: int = 32,
    num_workers: Optional[int] = os.cpu_count(),
) -> List[torch.utils.data.DataLoader]:
    """
    Crete a list of dataloaders for each dataset

    Args:
        root: list of paths to the dataset root
        annFile: list of paths to the annotation file
        shuffle: list of bools to shuffle the dataset
        resize_size: size to resize the images
        batch_size: batch size
        num_workers: number of workers to use for data loading

    Returns:
        list of dataloaders
    """
    if isinstance(root, str):
        root = [root]
    if isinstance(annFile, str):
        annFile = [annFile]
    if isinstance(shuffle, bool):
        shuffle = [shuffle] * len(root)

    if not (len(root) == len(annFile) == len(shuffle)):
        raise ValueError("root, annFile and shuffle must have the same length")

    dataloader_list: List[DataLoader] = []
    for root_tmp, annFile_tmp, shuffle_tmp in zip(root, annFile, shuffle):
        coco_ds = EncordCocoDetection(
            root=root_tmp,
            annFile=annFile_tmp,
            transforms=CocoMaskRCNNTransforms(resize_size=resize_size),
        )

        coco_ds = coco_remove_images_without_annotations(coco_ds)

        dataloader_list.append(
            DataLoader(
                coco_ds,
                batch_size=batch_size,
                shuffle=shuffle_tmp,
                num_workers=num_workers if num_workers is not None else 8,
                collate_fn=maskrcnn_collate_fn,
            )
        )
    return dataloader_list


def filter_by_metric(coco_ds: EncordCocoDetection) -> List[int]:
    """
    Filter the dataset by the metric
    Args:
        coco_ds: dataset to filter

    Returns:
        list of indexes to keep
    """
    valid_ids = []
    for image, target, metadata in iter(coco_ds):
        if True:  # TODO - add your own metric filter
            valid_ids.append(target[0]["image_id"])
    return valid_ids


def get_filtered_dataloader(
    root: str,
    annFile: str,
    *,
    resize_size: int = 512,
    batch_size: int = 32,
    num_workers: Optional[int] = os.cpu_count(),
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Crete a list of dataloaders for each dataset

    Args:
        root: list of paths to the dataset root
        annFile: list of paths to the annotation file
        resize_size: size to resize the images
        batch_size: batch size
        num_workers: number of workers to use for data loading

    Returns:
        list of dataloaders
    """

    coco_ds = EncordCocoDetection(
        root=root,
        annFile=annFile,
        transforms=CocoMaskRCNNTransforms(resize_size=resize_size),
    )

    coco_ds = coco_remove_images_without_annotations(coco_ds)

    filtered_idx = filter_by_metric(coco_ds)
    coco_filtered: torch.utils.data.Dataset = torch.utils.data.Subset(coco_ds, filtered_idx)

    dataloader_full: torch.utils.data.dataloader.DataLoader = DataLoader(
        coco_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else 8,
        collate_fn=maskrcnn_collate_fn,
        pin_memory=True,
    )

    dataloader_filtered: torch.utils.data.dataloader.DataLoader = DataLoader(
        coco_filtered,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else 8,
        collate_fn=maskrcnn_collate_fn,
        pin_memory=True,
    )

    return dataloader_full, dataloader_filtered
