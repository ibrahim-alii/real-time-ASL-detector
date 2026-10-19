from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from .contracts import DirectTranslationResult, TranslationRequest
from .schema import MMPOSE_SCHEMA, translation_schema

T5_MODEL_ID = "google/t5-v1_1-base"
T5_REVISION = "b5fc947a416ea3cb079532cb3c2bbadeb7f800fc"


@dataclass(frozen=True, slots=True)
class DirectTranslationConfig:
    pretrained_model: str = T5_MODEL_ID
    pretrained_revision: str = T5_REVISION
    decoder_max_tokens: int = 128
    beam_width: int = 5
    feature_schema: str = MMPOSE_SCHEMA
    input_encoding: str = "sentinel-v1"

    def __post_init__(self) -> None:
        if self.input_encoding not in {"sentinel-v1", "mask-v1"}:
            raise ValueError("unsupported translation input encoding")


def encoded_values(request: TranslationRequest, encoding: str) -> torch.Tensor:
    values = request.values.copy()
    if encoding == "mask-v1":
        schema = translation_schema(request.feature_schema)
        coordinates = values.reshape(len(values), schema.landmarks, -1)
        coordinates[~request.landmark_mask] = 0
        values = np.concatenate(
            (coordinates.reshape(len(values), -1), request.landmark_mask.astype(np.float32)), axis=1
        )
    elif encoding != "sentinel-v1":
        raise ValueError("unsupported translation input encoding")
    return torch.from_numpy(values)


class LandmarkT5(nn.Module):
    """T5.1.1 encoder-decoder with a schema-specific landmark projection."""

    def __init__(self, config: DirectTranslationConfig | None = None) -> None:
        super().__init__()
        try:
            from transformers import T5ForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("install asl-recognizer[hpc] to use pretrained T5.1.1") from error
        config = config or DirectTranslationConfig()
        self.config = config
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            config.pretrained_model, revision=config.pretrained_revision
        )
        dimension = translation_schema(config.feature_schema).dimension
        if config.input_encoding == "mask-v1":
            dimension += translation_schema(config.feature_schema).landmarks
        self.input_projection = nn.Linear(dimension, self.t5.config.d_model)

    def forward(
        self, values: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.t5(
                inputs_embeds=self.input_projection(values),
                attention_mask=attention_mask,
                labels=labels,
            ).loss,
        )

    @torch.inference_mode()
    def translate(
        self, request: TranslationRequest, tokenizer: object, *, device: str = "cpu"
    ) -> DirectTranslationResult:
        started = time.perf_counter()
        if request.feature_schema != self.config.feature_schema:
            raise ValueError("input feature schema does not match the trained model")
        values = encoded_values(request, self.config.input_encoding).unsqueeze(0).to(device)
        mask = torch.ones((1, len(request.values)), dtype=torch.long, device=device)
        self.to(device).eval()
        token_ids: Any = self.t5.generate(
            inputs_embeds=self.input_projection(values),
            attention_mask=mask,
            num_beams=self.config.beam_width,
            max_new_tokens=self.config.decoder_max_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        decode = tokenizer.decode  # type: ignore[attr-defined]
        english = str(decode(token_ids.sequences[0], skip_special_tokens=True)).strip()
        score_values = getattr(token_ids, "sequences_scores", None)
        score = float(score_values[0]) if score_values is not None else None
        return DirectTranslationResult(
            english=english,
            decoder_score=score,
            model_id=self.config.pretrained_model,
            model_version=str(self.t5.config._name_or_path),
            latency_ms=(time.perf_counter() - started) * 1000,
            beam_width=self.config.beam_width,
        )
