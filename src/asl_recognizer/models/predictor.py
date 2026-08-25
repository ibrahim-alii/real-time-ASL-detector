from __future__ import annotations

import torch

from ..features.schema import FeatureSequence
from ..streaming import Prediction
from .gru import LandmarkGRU


class GRUPredictor:
    """Adapt a trained GRU to the streaming probability contract."""

    def __init__(self, model: LandmarkGRU, labels: tuple[str, ...]) -> None:
        if len(labels) != model.config.num_classes or len(labels) != len(set(labels)):
            raise ValueError("labels must be unique and match the model class count")
        if any(not label.strip() for label in labels):
            raise ValueError("labels cannot be blank")
        self.model = model.eval()
        self.labels = labels

    def predict(self, sequence: FeatureSequence) -> Prediction:
        device = next(self.model.parameters()).device
        values = torch.from_numpy(sequence.values).unsqueeze(0).to(device)
        landmark_mask = torch.from_numpy(sequence.landmark_mask).unsqueeze(0).to(device)
        temporal_mask = torch.from_numpy(sequence.temporal_mask).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = self.model(values, landmark_mask, temporal_mask)
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return Prediction(dict(zip(self.labels, probabilities, strict=True)))

