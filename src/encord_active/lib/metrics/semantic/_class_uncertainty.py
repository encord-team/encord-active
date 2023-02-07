import logging
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import LeakyReLU

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.embeddings.cnn import get_cnn_embeddings
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    EmbeddingType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        layers=3,
        residual=0,
        dropout=None,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        if residual > 0:
            assert input_dim == out_dim, "Input and output dimensions must match for residual connections"

        self.residual = residual
        step = int(round((out_dim - input_dim) / layers))

        modules = []
        for _ in range(layers - 1):
            modules.append(nn.Linear(input_dim, input_dim + step))
            modules.append(activation(inplace=True))
            if dropout:
                modules.append(nn.Dropout(dropout))
            input_dim = input_dim + step
        modules.append(nn.Linear(input_dim, out_dim))
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        output = self.module(x)
        return self.residual * x + (1 - self.residual) * output if self.residual > 0 else output


class BayesianClassifier(nn.Module):
    def __init__(self, n_classes, n_train_samples, n_test_samples):
        super().__init__()
        self.n_samples = n_train_samples
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples

        self.classifier = nn.Sequential(
            DenseBlock(512, n_classes, dropout=0.1, activation=LeakyReLU),
            nn.Softmax(dim=-1),
        )

    def sample(self, embedding):
        outputs = []
        for _ in range(self.n_samples):
            outputs.append(self.classifier(embedding))
        return torch.stack(outputs, dim=-1)

    def forward(self, embedding):
        outputs = self.sample(embedding)
        return outputs

    @contextmanager
    def mc_eval(self):
        """Switch to evaluation mode with MC Dropout active."""
        istrain_classifier = self.classifier.training
        try:
            self.n_samples = self.n_test_samples
            self.classifier.eval()
            # Keep dropout active
            for m in self.classifier.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
            yield self.classifier
        finally:
            self.n_samples = self.n_train_samples
            if istrain_classifier:
                self.classifier.train()


def get_prediction_statistics(outputs):
    if len(outputs.shape) < 3:
        outputs = outputs.unsqueeze(-1)
    mean_probs = torch.mean(outputs, dim=-1)
    entropy = -torch.sum(torch.log(mean_probs) * mean_probs, dim=-1)
    model_preds = torch.argmax(mean_probs, dim=1)
    return model_preds, mean_probs, entropy


def train_model(classifier, model_path, batches, idx_to_counts, name_to_idx):
    train_batches, val_batches = train_test_split(batches)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=1 / torch.tensor(list(idx_to_counts.values()))).to(DEVICE)
    count = 0
    min_loss = torch.inf
    for epoch in range(50):
        classifier.train()

        train_loss_epoch = 0
        train_accuracy_epoch = 0
        for batch in train_batches:
            optimizer.zero_grad()
            labels, mean_probs, model_preds, entropy = base_predict(classifier, batch, name_to_idx)

            loss = criterion(mean_probs, labels)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() / len(train_batches)
            train_accuracy_epoch += (model_preds == labels).sum().item() / (len(train_batches) * len(labels))

        with classifier.mc_eval():
            val_loss_epoch = 0
            val_accuracy_epoch = 0
            for batch in val_batches:
                labels, mean_probs, model_preds, entropy = base_predict(classifier, batch, name_to_idx)

                loss = criterion(mean_probs, labels)
                val_loss_epoch += loss.item() / len(val_batches)
                val_accuracy_epoch += (model_preds == labels).sum().item() / (len(val_batches) * len(labels))

        if val_loss_epoch < min_loss:
            count = 0
            min_loss = val_loss_epoch
            torch.save(classifier.state_dict(), model_path)
        else:
            count += 1
            if count == 10:
                break

        logger.debug(f"---------------------------Epoch {epoch} ---------------------------")
        logger.debug(f"Train loss {train_loss_epoch} || Val loss {val_loss_epoch}")
        logger.debug(f"Train accuracy {train_accuracy_epoch} || Val accuracy {val_accuracy_epoch}")


def base_predict(classifier, batch, name_to_idx):
    embeddings = torch.Tensor([eval(x) for x in batch["embedding"]]).to(DEVICE)
    labels = torch.LongTensor([name_to_idx[x] for x in batch["object_class"]]).to(DEVICE)
    outputs = classifier(embeddings)
    model_preds, mean_probs, entropy = get_prediction_statistics(outputs)
    return labels, mean_probs, model_preds, entropy


def train_test_split(batches):
    sizes = [int(0.75 * len(batches)), int(0.25 * len(batches))]
    sizes = [max(1, x) for x in sizes]
    cumsum = np.cumsum(sizes)
    cumsum = np.insert(cumsum, 0, 0)
    train_batches, val_batches = [batches[cumsum[i] : cumsum[i + 1]] for i in range(len(sizes))]
    return train_batches, val_batches


def get_batches_and_model(resnet_embeddings_df):
    cls_set = set(resnet_embeddings_df["object_class"])
    name_to_idx = {name: idx for idx, name in enumerate(cls_set)}
    idx_to_counts = {
        name_to_idx[cls_name]: (resnet_embeddings_df["object_class"] == cls_name).sum() for cls_name in cls_set
    }
    classifier = BayesianClassifier(n_classes=len(cls_set), n_train_samples=3, n_test_samples=20).to(DEVICE)
    batches = np.array_split(resnet_embeddings_df, resnet_embeddings_df.shape[0] // 64)
    return batches, classifier, idx_to_counts, name_to_idx


def preliminaries(iterator):
    model_path = os.path.join(iterator.cache_dir, "models", f"{Path(__file__).stem}_classifier.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    resnet_embeddings_df = get_cnn_embeddings(iterator, embedding_type=EmbeddingType.OBJECT, force=False)
    batches, classifier, idx_to_counts, name_to_idx = get_batches_and_model(resnet_embeddings_df)
    if not os.path.isfile(model_path):
        train_model(classifier, model_path, batches, idx_to_counts, name_to_idx)
    classifier.load_state_dict(torch.load(model_path))
    return batches, classifier, name_to_idx, resnet_embeddings_df


class EntropyMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Class Entropy",
            short_description="Estimates the uncertainty of the assigned label through the distribution entropy",
            long_description=r"""Uses the entropy of the distribution over labels from a lightweight classifier neural
network and Monte-Carlo Dropout to estimate the uncertainty of the label.""",
            metric_type=MetricType.SEMANTIC,
            data_type=DataType.IMAGE,
            annotation_type=[AnnotationType.OBJECT.BOUNDING_BOX, AnnotationType.OBJECT.POLYGON],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        batches, classifier, name_to_idx, resnet_embeddings_df = preliminaries(iterator)
        with classifier.mc_eval() and torch.inference_mode():
            pbar = tqdm.tqdm(total=len(resnet_embeddings_df), desc="Predicting uncertainty")
            for batch in batches:
                labels, mean_probs, model_preds, entropy = base_predict(classifier, batch, name_to_idx)

                for i, ent in enumerate(entropy):
                    writer.write(
                        round(ent.item(), 4),
                        key=batch["identifier"].iloc[i],
                        description=f"Assigned class is {batch['object_class'].iloc[i]}.",
                        label_class=batch["object_class"].iloc[i],
                        url=batch["url"].iloc[i],
                        frame=batch["frame"].iloc[i],
                    )
                    pbar.update(1)


class ConfidenceScoreMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Class Confidence Score",
            short_description="Estimates the confidence of the assigned label.",
            long_description=r"""Estimates the confidence of the assigned label as the probability of the assigned label.""",
            metric_type=MetricType.SEMANTIC,
            data_type=DataType.IMAGE,
            annotation_type=[AnnotationType.OBJECT.BOUNDING_BOX, AnnotationType.OBJECT.POLYGON],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        batches, classifier, name_to_idx, resnet_embeddings_df = preliminaries(iterator)

        with classifier.mc_eval() and torch.inference_mode():
            pbar = tqdm.tqdm(total=len(resnet_embeddings_df), desc="Predicting uncertainty")
            for batch in batches:
                labels, mean_probs, model_preds, entropy = base_predict(classifier, batch, name_to_idx)
                model_confidence = torch.gather(mean_probs, dim=-1, index=labels.unsqueeze(-1))
                for i, conf in enumerate(model_confidence):
                    writer.write(
                        round(conf.item(), 6),
                        key=batch["identifier"].iloc[i],
                        description=f"Assigned class is {batch['object_class'].iloc[i]} with confidence {round(conf.item(), 6)}.",
                        label_class=batch["object_class"].iloc[i],
                        url=batch["url"].iloc[i],
                        frame=batch["frame"].iloc[i],
                    )
                    pbar.update(1)
