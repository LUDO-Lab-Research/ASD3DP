#!/usr/bin/env python
"""Chunked ASD3DP feature cache for source-only AE experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FeatureConfig:
    n_mels: int = 128
    frames: int = 5
    frame_hop_length: int = 1
    n_fft: int = 1024
    hop_length: int = 512
    power: float = 2.0
    fmin: float = 0.0
    fmax: float | None = None
    win_length: int | None = None

    @property
    def dims(self) -> int:
        return self.n_mels * self.frames

    @property
    def cache_name(self) -> str:
        fmax = "none" if self.fmax is None else str(self.fmax)
        win = "none" if self.win_length is None else str(self.win_length)
        return (
            f"mels{self.n_mels}_frames{self.frames}_fh{self.frame_hop_length}_"
            f"fft{self.n_fft}_hop{self.hop_length}_p{self.power:g}_"
            f"fmin{self.fmin:g}_fmax{fmax}_win{win}"
        )


def iter_wavs(raw_root: Path, split: str, limit: int | None = None) -> list[Path]:
    files = sorted((raw_root / split).glob("*.wav"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no wav files found: {raw_root / split}")
    return files


def label_from_name(path: Path) -> int:
    if "_anomaly_" in path.name:
        return 1
    if "_normal_" in path.name:
        return 0
    raise ValueError(f"cannot infer label from filename: {path}")


def waveform_to_vectors(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    if getattr(y, "ndim", 1) > 1:
        y = librosa.to_mono(y)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=cfg.power,
        fmax=cfg.fmax,
        fmin=cfg.fmin,
        win_length=cfg.win_length,
    )
    log_mel = 20.0 / cfg.power * np.log10(np.maximum(mel, sys.float_info.epsilon))
    n_vectors = len(log_mel[0, :]) - cfg.frames + 1
    if n_vectors < 1:
        return np.empty((0, cfg.dims), dtype=np.float32)
    vectors = np.zeros((n_vectors, cfg.dims), dtype=np.float32)
    for frame_idx in range(cfg.frames):
        start = cfg.n_mels * frame_idx
        stop = cfg.n_mels * (frame_idx + 1)
        src_start = frame_idx * cfg.frame_hop_length
        src_stop = src_start + n_vectors
        vectors[:, start:stop] = log_mel[:, src_start:src_stop].T
    return vectors


def file_to_vectors(path: Path, cfg: FeatureConfig) -> np.ndarray:
    y, sr = librosa.load(path, sr=None, mono=False)
    return waveform_to_vectors(y, sr, cfg)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "split",
        "file_index",
        "basename",
        "path",
        "label",
        "chunk",
        "chunk_row_start",
        "row_count",
        "global_row_start",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_feature_store(
    raw_root: Path,
    cache_root: Path,
    split: str,
    cfg: FeatureConfig,
    chunk_files: int = 256,
    limit_files: int | None = None,
    force: bool = False,
) -> Path:
    out_dir = cache_root / cfg.cache_name / split
    done_path = out_dir / "DONE.json"
    manifest_path = out_dir / "manifest.csv"
    if done_path.is_file() and manifest_path.is_file() and not force:
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chunk_*.npy"):
        old.unlink()
    manifest_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)

    files = iter_wavs(raw_root, split, limit_files)
    rows: list[dict[str, object]] = []
    global_row_start = 0
    chunk_id = 0

    for start in range(0, len(files), chunk_files):
        batch_files = files[start : start + chunk_files]
        chunk_arrays: list[np.ndarray] = []
        chunk_row_start = 0
        for offset, wav_path in enumerate(batch_files):
            file_index = start + offset
            vectors = file_to_vectors(wav_path, cfg)
            row_count = int(vectors.shape[0])
            if row_count:
                chunk_arrays.append(vectors)
            rows.append(
                {
                    "split": split,
                    "file_index": file_index,
                    "basename": wav_path.name,
                    "path": str(wav_path),
                    "label": label_from_name(wav_path),
                    "chunk": f"chunk_{chunk_id:06d}.npy",
                    "chunk_row_start": chunk_row_start,
                    "row_count": row_count,
                    "global_row_start": global_row_start,
                }
            )
            chunk_row_start += row_count
            global_row_start += row_count

        chunk = (
            np.concatenate(chunk_arrays, axis=0).astype(np.float32, copy=False)
            if chunk_arrays
            else np.empty((0, cfg.dims), dtype=np.float32)
        )
        np.save(out_dir / f"chunk_{chunk_id:06d}.npy", chunk)
        print(f"[{split}] wrote chunk_{chunk_id:06d}.npy files={len(batch_files)} rows={len(chunk)}")
        del chunk_arrays, chunk
        chunk_id += 1

    write_manifest(manifest_path, rows)
    done = {
        "config": asdict(cfg),
        "split": split,
        "files": len(files),
        "rows": global_row_start,
        "chunks": chunk_id,
        "chunk_files": chunk_files,
    }
    done_path.write_text(json.dumps(done, indent=2) + "\n")
    return out_dir


class ChunkedFeatureDataset(Dataset):
    def __init__(self, store_dir: Path, indices: np.ndarray | None = None) -> None:
        self.store_dir = Path(store_dir)
        with (self.store_dir / "manifest.csv").open(newline="") as f:
            self.manifest_rows = list(csv.DictReader(f))
        self.file_starts = np.array(
            [int(row["global_row_start"]) for row in self.manifest_rows], dtype=np.int64
        )
        self.file_counts = np.array(
            [int(row["row_count"]) for row in self.manifest_rows], dtype=np.int64
        )
        self.length = int(self.file_counts.sum())
        self.indices = (
            np.arange(self.length, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        done = json.loads((self.store_dir / "DONE.json").read_text())
        self.dims = int(done["config"]["n_mels"]) * int(done["config"]["frames"])
        self._chunk_cache_name: str | None = None
        self._chunk_cache: np.ndarray | None = None

    def __len__(self) -> int:
        return int(len(self.indices))

    def _load_chunk(self, chunk_name: str) -> np.ndarray:
        if self._chunk_cache_name != chunk_name:
            self._chunk_cache = np.load(self.store_dir / chunk_name, mmap_mode="r")
            self._chunk_cache_name = chunk_name
        assert self._chunk_cache is not None
        return self._chunk_cache

    def row(self, global_row: int) -> np.ndarray:
        file_idx = int(np.searchsorted(self.file_starts, global_row, side="right") - 1)
        row = self.manifest_rows[file_idx]
        local = int(row["chunk_row_start"]) + (global_row - int(row["global_row_start"]))
        return np.asarray(self._load_chunk(row["chunk"])[local], dtype=np.float32).copy()

    def __getitem__(self, item: int) -> torch.Tensor:
        return torch.from_numpy(self.row(int(self.indices[item])))

    def file_score_table(self, row_scores: np.ndarray) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for manifest_row in self.manifest_rows:
            start = int(manifest_row["global_row_start"])
            count = int(manifest_row["row_count"])
            score = math.nan if count == 0 else float(np.mean(row_scores[start : start + count]))
            rows.append(
                {
                    "basename": manifest_row["basename"],
                    "path": manifest_row["path"],
                    "label": int(manifest_row["label"]),
                    "score": score,
                    "row_count": count,
                }
            )
        return rows


def split_train_validation(
    dataset: ChunkedFeatureDataset,
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(dataset.length, dtype=np.int64)
    rng.shuffle(all_indices)
    n_val = int(round(len(all_indices) * validation_split))
    if n_val < 1:
        return np.sort(all_indices), np.empty(0, dtype=np.int64)
    return np.sort(all_indices[n_val:]), np.sort(all_indices[:n_val])


def prepare_cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], action="append", required=True)
    parser.add_argument("--chunk-files", type=int, default=256)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = FeatureConfig()
    for split in args.split:
        build_feature_store(
            args.raw_root,
            args.cache_root,
            split,
            cfg,
            chunk_files=args.chunk_files,
            limit_files=args.limit_files,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(prepare_cli())
