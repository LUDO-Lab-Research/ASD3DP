"""Embedding-space Residual View for paired near/far features.

The filename keeps the original project spelling: `differet_view.py`.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

import torch

T = TypeVar("T")

DIFFERENT_VIEW_NAMES = {"fixed_residual_view"}
SCALED_DIFFERENT_VIEW_NAMES = {"fixed_residual_view"}
PAIR_INPUT_TYPES = ["near", "far"]


def make_different_view_features(
    feature_layers: torch.Tensor,
    different_view: str,
    fixed_residual_alpha: float | None = None,
) -> torch.Tensor:
    """Create Residual View features from alternating near/far embeddings.

    Expected shape is `(layers, 2 * n_clips, dim)`, where each clip appears as
    `near` then `far`.
    """
    if different_view != "fixed_residual_view":
        raise ValueError(
            f"Unknown different_view={different_view}. "
            "This public reproduction package supports only fixed_residual_view."
        )
    if fixed_residual_alpha is None:
        raise ValueError("fixed_residual_view requires fixed_residual_alpha.")
    if feature_layers.ndim != 3:
        raise ValueError(
            "fixed_residual_view expects flattened layer features with shape "
            f"(layers, samples, dim), got shape={tuple(feature_layers.shape)}."
        )
    if feature_layers.shape[1] % 2 != 0:
        raise ValueError(
            "fixed_residual_view expects paired near/far samples, got odd "
            f"sample count={feature_layers.shape[1]}."
        )

    near = feature_layers[:, 0::2]
    far = feature_layers[:, 1::2]
    return near - fixed_residual_alpha * far


def collapse_paired_sequence(values: Sequence[T]) -> list[T]:
    """Keep one metadata entry per near/far pair."""
    if len(values) % 2 != 0:
        raise ValueError(
            f"Paired metadata length must be even, got {len(values)}."
        )
    return list(values[0::2])
