from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

from ..features.schema import FEATURE_DIMENSION, LANDMARK_COUNT


@dataclass(frozen=True, slots=True)
class GRUConfig:
    num_classes: int
    feature_dimension: int = FEATURE_DIMENSION
    landmark_count: int = LANDMARK_COUNT
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if self.feature_dimension < 1 or self.landmark_count < 1:
            raise ValueError("input dimensions must be positive")
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("GRU dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GRUConfig:
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise ValueError("invalid GRU configuration fields")
        return cls(**value)


class LandmarkGRU(nn.Module):
    """Compact GRU using coordinate and landmark-presence features."""

    def __init__(self, config: GRUConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config.feature_dimension + config.landmark_count,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(
        self, values: Tensor, landmark_mask: Tensor, temporal_mask: Tensor
    ) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.config.feature_dimension:
            raise ValueError("values must have shape [batch, time, feature_dimension]")
        if landmark_mask.shape != (*values.shape[:2], self.config.landmark_count):
            raise ValueError("landmark_mask has an incompatible shape")
        if temporal_mask.shape != values.shape[:2]:
            raise ValueError("temporal_mask has an incompatible shape")
        if not bool(temporal_mask.any(dim=1).all()):
            raise ValueError("each sequence must contain at least one valid timestep")

        inputs = torch.cat((values, landmark_mask.to(dtype=values.dtype)), dim=-1)
        outputs, _ = self.gru(inputs)
        weights = temporal_mask.unsqueeze(-1).to(dtype=outputs.dtype)
        pooled = (outputs * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))
