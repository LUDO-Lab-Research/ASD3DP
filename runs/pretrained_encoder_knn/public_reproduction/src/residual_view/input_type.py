"""Near/far waveform input selection for DCASE2026 two-channel audio."""

from __future__ import annotations

import numpy as np

INPUT_TYPE_NAMES = {"near", "far"}


def make_input_type(wave: np.ndarray, input_type: str) -> np.ndarray:
    """Select one mono waveform from a 2-channel DCASE2026 recording."""
    if wave.ndim != 2 or wave.shape[1] < 2:
        raise ValueError(
            f"Input type {input_type} requires 2-channel audio, "
            f"got shape={wave.shape}."
        )
    if input_type == "near":
        return wave[:, 0]
    if input_type == "far":
        return wave[:, 1]
    raise ValueError(
        f"Unknown input_type={input_type}. Expected one of {sorted(INPUT_TYPE_NAMES)}."
    )


def expand_input_type(input_type: str, split_role: str) -> list[str]:
    """Return the channel inputs needed for one run.

    Residual View extraction requests both near and far internally; plain runs
    use the requested single channel.
    """
    if input_type not in INPUT_TYPE_NAMES:
        raise ValueError(
            f"Unknown input_type={input_type}. Expected one of {sorted(INPUT_TYPE_NAMES)}."
        )
    if split_role not in {"train", "eval"}:
        raise ValueError(
            f"Unknown split_role={split_role}. Expected 'train' or 'eval'."
        )
    return [input_type]
