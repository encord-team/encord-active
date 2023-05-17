import os
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import tqdm
from torch import nn
from torchvision.io import read_image
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.semantic._class_uncertainty import train_test_split
from encord_active.lib.metrics.writer import CSVMetricWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegmentationHead(nn.Module):
    def __init__(self, n_classes: int, n_train_samples: int, n_test_samples: int):
        super().__init__()
        self.n_samples = n_train_samples
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples

        self.backbone = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.heads = nn.Sequential(
            nn.Conv2d(21, 21 + (n_classes - 21) // 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(21 + (n_classes - 21) // 2, n_classes, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, img):
        with torch.no_grad():
            embedding = self.backbone(img)["out"]
        x = self.sample(embedding)
        return x

    def sample(self, embedding):
        outputs = []
        for _ in range(self.n_samples):
            outputs.append(self.heads(embedding))
        return torch.stack(outputs, dim=-1)

    @contextmanager
    def mc_eval(self):
        """Switch to evaluation mode with MC Dropout active."""
        istrain_head = self.heads.training
        try:
            self.n_samples = self.n_test_samples
            self.eval()

            # Keep dropout active
            for m in self.heads.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
            yield self
        finally:
            self.n_samples = self.n_train_samples
            if istrain_head:
                self.heads.train()
                self.backbone.eval()


def get_batches(iterator, batch_size=6):
    width = height = 224
    transforms = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
    transforms.resize_size = [width, height]
    name_to_idx = {"background": 0}
    idx_to_counts = {0: 0}
    image_list, mask_list, key_list = [], [], []
    for du, img_pth in iterator.iterate(desc="Loading data"):
        img_mask = np.zeros((height, width), dtype=np.uint8)

        for obj in du["labels"].get("objects", []):
            if obj["shape"] != "polygon":
                continue

            cls_name = obj["name"]
            if cls_name not in name_to_idx:
                name_to_idx[cls_name] = len(name_to_idx)
            cls_idx: int = name_to_idx[cls_name]

            p = obj["polygon"]
            polygon = np.array(
                [(p[str(i)]["x"] * width, p[str(i)]["y"] * height) for i in range(len(p))],
                dtype=np.int32,
            )

            img_mask = cv2.fillPoly(img_mask, [polygon], cls_idx)

            if cls_idx not in idx_to_counts:
                idx_to_counts[cls_idx] = (img_mask == cls_idx).sum()
            else:
                idx_to_counts[cls_idx] += (img_mask == cls_idx).sum()

        idx_to_counts[0] += (img_mask == 0).sum()

        img_tensor = read_image(img_pth.as_posix()) / 255

        if img_tensor.shape[0] == 4:
            img_tensor = img_tensor[:3]

        if img_tensor.shape[0] < 3:
            for i in range(3 - img_tensor.shape[0]):
                img_tensor = torch.cat([img_tensor, img_tensor[-1:]], dim=0)

        image_list.append(transforms(img_tensor))
        mask_list.append(torch.LongTensor(img_mask))
        key_list.append(iterator.get_identifier())

    img_batches = [torch.stack(image_list[i : i + batch_size], dim=0) for i in range(0, len(image_list), batch_size)]
    mask_batches = [torch.stack(mask_list[i : i + batch_size], dim=0) for i in range(0, len(mask_list), batch_size)]
    key_batches = [key_list[i : i + batch_size] for i in range(0, len(key_list), batch_size)]
    return list(zip(img_batches, mask_batches, key_batches)), idx_to_counts, name_to_idx


def get_prediction_statistics(logits):
    if len(logits.shape) < 5:
        logits = logits.unsqueeze(-1)

    mean_logits = logits.mean(dim=-1)
    mean_probs = torch.softmax(mean_logits, dim=1)
    entropy_map = -(torch.log(mean_probs) * mean_probs).sum(1)
    model_preds = mean_probs.argmax(dim=1)
    return model_preds, mean_logits, entropy_map


def train_model(model, model_path, batches):
    train_batches, val_batches = train_test_split(batches)
    optimizer = torch.optim.Adam(model.heads.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    count = 0
    min_loss = torch.inf
    for epoch in range(50):
        model.backbone.eval()
        model.heads.train()

        train_loss_epoch = 0
        train_accuracy_epoch = 0
        for imgs, masks, keys in train_batches:
            optimizer.zero_grad()
            labels = masks.to(DEVICE)
            logits = model(imgs.to(DEVICE)).squeeze()
            model_preds, mean_logits, entropy_map = get_prediction_statistics(logits)
            loss = criterion(mean_logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() / len(train_batches)
            train_accuracy_epoch += (model_preds == labels).sum().item() / (len(train_batches) * labels.nelement())

        with torch.inference_mode():
            with model.mc_eval():
                val_loss_epoch = 0
                val_accuracy_epoch = 0
                for imgs, masks, keys in val_batches:
                    labels = masks.to(DEVICE)
                    logits = model(imgs.to(DEVICE)).squeeze()
                    model_preds, mean_logits, entropy_map = get_prediction_statistics(logits)
                    loss = criterion(mean_logits, labels)
                    val_loss_epoch += loss.item() / len(val_batches)
                    val_accuracy_epoch += (model_preds == labels).sum().item() / (len(val_batches) * labels.nelement())

        scheduler.step()

        if val_loss_epoch < min_loss:
            count = 0
            min_loss = val_loss_epoch
            torch.save(model.heads.state_dict(), model_path)
        else:
            count += 1
            if count == 10:
                break

        print(f"---------------------------Epoch {epoch} ---------------------------")
        print(f"Train loss {train_loss_epoch} || Val loss {val_loss_epoch}")
        print(f"Train accuracy {train_accuracy_epoch} || Val accuracy {val_accuracy_epoch}")


class EntropyHeatmapMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Heatmap uncertainty",
            short_description="Estimates the uncertainty of the segmentation through the entropy of the pixelwise class distribution",
            long_description=r"""""",
            metric_type=MetricType.SEMANTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.ALL,
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        model_path = os.path.join(iterator.cache_dir, "models", f"{Path(__file__).stem}_model.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        batches, idx_to_counts, name_to_idx = get_batches(iterator)

        model = SegmentationHead(n_classes=len(idx_to_counts), n_train_samples=1, n_test_samples=10).to(DEVICE)

        if not os.path.isfile(model_path):
            train_model(model, model_path, batches)
        model.heads.load_state_dict(torch.load(model_path))

        with torch.inference_mode():
            with model.mc_eval():
                pbar = tqdm.tqdm(total=len(batches), desc="Predicting uncertainty")
                for imgs, masks, keys in batches:
                    logits = model(imgs.to(DEVICE)).squeeze()
                    model_preds, mean_logits, entropy_map = get_prediction_statistics(logits)

                    for i, (k, ent) in enumerate(zip(keys, entropy_map)):
                        writer.write(
                            round(ent.mean().item(), 4),
                            key=k,  # remove key somehow
                            description=str(ent.type(torch.HalfTensor).cpu().tolist()),
                            url="",
                        )
                    pbar.update(1)
