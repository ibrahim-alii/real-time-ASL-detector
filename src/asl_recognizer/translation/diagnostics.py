"""Bounded conditioning diagnostics: teacher likelihood, controls, gradients and input cost."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from .cache import TranslationFeatureCache
from .contracts import TranslationRequest
from .evaluation import evaluate_translations
from .model import encoded_values


def derangement(count: int, seed: int) -> list[int]:
    if count < 2:
        raise ValueError("conditioning requires at least two examples")
    rng = np.random.default_rng(seed)
    for _ in range(128):
        order = rng.permutation(count).tolist()
        if all(index != value for index, value in enumerate(order)):
            return list(order)
    return [*range(1, count), 0]


def _loss(
    model: Any, tokenizer: Any, request: TranslationRequest, caption: str, device: str
) -> tuple[torch.Tensor, int, torch.Tensor]:
    values = encoded_values(request, model.config.input_encoding).unsqueeze(0).to(device)
    attention = torch.ones((1, len(request.values)), dtype=torch.long, device=device)
    tokens = tokenizer(
        [caption], padding=True, truncation=True, max_length=128, return_tensors="pt"
    )["input_ids"].to(device)
    labels = tokens.masked_fill(tokens == tokenizer.pad_token_id, -100)
    count = int(labels.ne(-100).sum())
    if not count:
        raise ValueError("conditioning target has no valid tokens")
    loss = model(values, attention, labels)
    if not bool(torch.isfinite(loss)):
        raise ValueError("conditioning likelihood is non-finite")
    return loss, count, values


def conditioning_report(
    model: Any,
    tokenizer: Any,
    examples: list[tuple[TranslationRequest, str]],
    *,
    device: str,
    wall_time_seconds: float = 900,
) -> dict[str, Any]:
    import sacrebleu

    if (not 2 <= len(examples) <= 32 or not math.isfinite(wall_time_seconds)
            or wall_time_seconds <= 0):
        raise ValueError("conditioning requires 2 to 32 examples and a positive time budget")
    deadline = time.monotonic() + wall_time_seconds * 0.8
    requests = [request for request, _ in examples]
    references = tuple(caption for _, caption in examples)
    orders = {f"permutation-{seed}": derangement(len(examples), seed) for seed in (17, 29, 43)}
    controls = {
        "real": requests,
        **{name: [requests[index] for index in order] for name, order in orders.items()},
        "static": [
            TranslationRequest(
                np.repeat(r.values[:1], len(r.values), axis=0),
                np.repeat(r.landmark_mask[:1], len(r.values), axis=0),
                r.feature_schema,
            )
            for r in requests
        ],
        "masked": [
            TranslationRequest(
                np.full_like(r.values, -1000), np.zeros_like(r.landmark_mask), r.feature_schema
            )
            for r in requests
        ],
    }
    was_training = model.training
    results: dict[str, Any] = {}
    model.eval()
    try:
        with torch.inference_mode():
            for name, inputs in controls.items():
                weighted_loss, count = 0.0, 0
                outputs = []
                per_example = []
                for request, caption in zip(inputs, references, strict=True):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("conditioning reserve reached")
                    loss, tokens, _ = _loss(model, tokenizer, request, caption, device)
                    weighted_loss += float(loss) * tokens
                    count += tokens
                    prediction = model.translate(request, tokenizer, device=device)
                    outputs.append(prediction)
                    per_example.append(
                        {
                            "reference": caption,
                            "prediction": prediction.english,
                            "target_nll": float(loss),
                            "tokens": tokens,
                        }
                    )
                metric = evaluate_translations(tuple(outputs), references)
                results[name] = {
                    "target_nll": weighted_loss / count,
                    "target_tokens": count,
                    "bleu": metric.bleu,
                    "chrf": float(
                        sacrebleu.corpus_chrf(
                            [p.english for p in outputs], [list(references)]
                        ).score
                    ),
                    "examples": per_example,
                }
        model.zero_grad(set_to_none=True)
        loss, _, values = _loss(model, tokenizer, requests[0], references[0], device)
        activation = model.input_projection(values).detach().float()
        activation_stats = {
            "mean": float(activation.mean()),
            "std": float(activation.std(unbiased=False)),
            "max_abs": float(activation.abs().max()),
        }
        torch.autograd.backward(loss)
        gradients = {}
        for group, prefix in {
            "input_projection": "input_projection.",
            "encoder": "t5.encoder.",
            "decoder": "t5.decoder.",
            "shared": "t5.shared.",
        }.items():
            gradients[group] = math.sqrt(
                sum(
                    float(parameter.grad.detach().float().square().sum())
                    for name, parameter in model.named_parameters()
                    if name.startswith(prefix) and parameter.grad is not None
                )
            )
        if any(not math.isfinite(value) for value in gradients.values()):
            raise ValueError("conditioning gradients are non-finite")
        return {
            "controls": results,
            "permutations": orders,
            "distinct_permutations": len({tuple(order) for order in orders.values()}),
            "projection_activation": activation_stats,
            "gradient_norms": gradients,
            "gradient_examples": 1,
            "parameter_updates": 0,
            "test_examples_used": 0,
            "semantic_review": "pending",
            "quality_validated": False,
            "likelihood_normalization": "valid-target-tokens-v1",
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def input_profile(
    cache: TranslationFeatureCache,
    clip_id: str,
    caption: str,
    tokenizer: Any,
    *,
    iterations: int = 20,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("input profile iterations must be positive")
    uncached = TranslationFeatureCache(cache.root, max_memory_bytes=0)
    timings = {}
    for name, loader in (("verified_disk_ms", uncached), ("warm_feature_ms", cache)):
        loader.load(clip_id)
        started = time.perf_counter()
        for _ in range(iterations):
            loader.load(clip_id)
        timings[name] = (time.perf_counter() - started) * 1000 / iterations
    started = time.perf_counter()
    for _ in range(iterations):
        tokenizer([caption], padding=True, truncation=True, max_length=128, return_tensors="pt")
    timings["tokenizer_ms"] = (time.perf_counter() - started) * 1000 / iterations
    return {
        **timings,
        "iterations": iterations,
        "feature_payload_bytes": cache.memory_bytes,
        "feature_payload_budget_bytes": cache.max_memory_bytes,
    }

