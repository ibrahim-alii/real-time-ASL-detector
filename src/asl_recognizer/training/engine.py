from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.gru import LandmarkGRU
from ..reproducibility import seed_everything
from .data import SequenceBatch, SequenceExample, collate_examples


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    validation_accuracy: tuple[float, ...]
    validation_macro_f1: tuple[float, ...]
    best_epoch: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    predictions: tuple[int, ...]
    labels: tuple[int, ...]


def _loader(
    examples: list[SequenceExample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader[SequenceExample]:
    if not examples:
        raise ValueError("examples cannot be empty")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_examples,
    )


def _loss_for_batch(
    model: LandmarkGRU, batch: SequenceBatch, criterion: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(batch.values, batch.landmark_mask, batch.temporal_mask)
    return criterion(logits, batch.labels), logits


def train_model(
    model: LandmarkGRU,
    train_examples: list[SequenceExample],
    validation_examples: list[SequenceExample],
    config: TrainingConfig = TrainingConfig(),
) -> TrainingHistory:
    seed_everything(config.seed)
    device = torch.device(config.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader = _loader(
        train_examples, batch_size=config.batch_size, shuffle=True, seed=config.seed
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []
    validation_macro_f1: list[float] = []
    best_macro_f1 = -1.0
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale_epochs = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss_for_batch(model, batch, criterion)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_losses.append(float(np.mean(epoch_losses)))

        validation = evaluate(
            model, validation_examples, batch_size=config.batch_size, device=config.device
        )
        validation_losses.append(validation.loss)
        validation_accuracies.append(validation.accuracy)
        validation_macro_f1.append(validation.macro_f1)
        improved = validation.macro_f1 > best_macro_f1 or (
            validation.macro_f1 == best_macro_f1 and validation.loss < best_loss
        )
        if improved:
            best_macro_f1 = validation.macro_f1
            best_loss = validation.loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return TrainingHistory(
        tuple(train_losses),
        tuple(validation_losses),
        tuple(validation_accuracies),
        tuple(validation_macro_f1),
        best_epoch,
    )


def evaluate(
    model: LandmarkGRU,
    examples: list[SequenceExample],
    *,
    batch_size: int = 32,
    device: str = "cpu",
) -> EvaluationResult:
    loader = _loader(examples, batch_size=batch_size, shuffle=False)
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    predictions: list[int] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(target_device)
            loss, logits = _loss_for_batch(model, batch, criterion)
            total_loss += float(loss.cpu())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch.labels.cpu().tolist())
    correct = sum(
        prediction == label for prediction, label in zip(predictions, labels, strict=True)
    )
    f1_scores: list[float] = []
    for class_id in range(model.config.num_classes):
        true_positive = sum(
            prediction == class_id and label == class_id
            for prediction, label in zip(predictions, labels, strict=True)
        )
        false_positive = sum(
            prediction == class_id and label != class_id
            for prediction, label in zip(predictions, labels, strict=True)
        )
        false_negative = sum(
            prediction != class_id and label == class_id
            for prediction, label in zip(predictions, labels, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator else 0.0)
    return EvaluationResult(
        loss=total_loss / len(labels),
        accuracy=correct / len(labels),
        macro_f1=float(np.mean(f1_scores)),
        predictions=tuple(predictions),
        labels=tuple(labels),
    )
