#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def channel_path(front_path: Path, channel: str) -> Path:
    value = str(front_path)
    if "_ch2_" not in value:
        raise ValueError(f"front ch2 marker not found: {front_path}")
    return Path(value.replace("_ch2_", f"_{channel}_"))


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    wave, sr = sf.read(path, always_2d=True, dtype="float32")
    if wave.shape[1] != 1:
        raise ValueError(f"expected mono channel file: {path} shape={wave.shape}")
    return wave[:, 0], int(sr)


def resample(wave: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return wave.astype(np.float32, copy=False)
    if source_sr % target_sr != 0:
        raise ValueError(f"unsupported sample-rate ratio: {source_sr} -> {target_sr}")
    return resample_poly(wave, target_sr, source_sr).astype(np.float32, copy=False)


def write_pair(rear_path: Path, front_path: Path, output: Path, target_sr: int = 16000) -> None:
    rear, rear_sr = load_mono(rear_path)
    front, front_sr = load_mono(front_path)
    rear = resample(rear, rear_sr, target_sr)
    front = resample(front, front_sr, target_sr)
    if len(rear) != len(front):
        raise ValueError(
            f"paired channel length mismatch: rear={len(rear)} front={len(front)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, np.column_stack([rear, front]), target_sr, subtype="PCM_16")


def stage_split(
    manifest: Path,
    release_root: Path,
    output_dir: Path,
    split: str,
    near_channel: str,
) -> list[dict]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    staged = []
    for index, row in enumerate(rows):
        if row["channel"] != "ch2":
            raise ValueError(f"Run 21 source manifest must be front ch2: {row['clip_uid']}")
        front_rel = Path(row["clip_audio_file"])
        rear_rel = channel_path(front_rel, near_channel)
        front = release_root / front_rel
        rear = release_root / rear_rel
        if not front.is_file() or not rear.is_file():
            raise FileNotFoundError(f"missing pair for {row['clip_uid']}: {rear} | {front}")
        name = f"section_00_{split}_{row['benchmark_label']}_{index:04d}_{row['clip_uid']}.wav"
        destination = output_dir / split / name
        write_pair(rear, front, destination)
        staged.append(
            {
                "index": index,
                "clip_uid": row["clip_uid"],
                "label": row["benchmark_label"],
                "fault_mode": row["fault_mode"],
                "near_source": str(rear),
                "front_far_source": str(front),
                "staged_file": str(destination),
            }
        )
    return staged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--near-channel", choices=["ch1", "ch3", "ch4"], default="ch1")
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Replace an existing staging directory. Without this flag the command fails closed.",
    )
    args = parser.parse_args()
    if args.output_root.exists():
        if not args.replace_output:
            raise FileExistsError(
                f"staging output already exists: {args.output_root}; "
                "pass --replace-output to rebuild it"
            )
        shutil.rmtree(args.output_root)
    train = stage_split(
        args.train_manifest, args.release_root, args.output_root, "train", args.near_channel
    )
    test = stage_split(
        args.test_manifest, args.release_root, args.output_root, "test", args.near_channel
    )
    if (len(train), len(test)) != (1000, 600):
        raise RuntimeError(f"unexpected split sizes: train={len(train)} test={len(test)}")
    audit = {
        "status": "STAGED_NEAR_FRONT_FAR",
        "near_channel": args.near_channel,
        "far_channel": "ch2",
        "channel_order": [f"{args.near_channel}_near", "front_ch2_far"],
        "sample_rate_hz": 16000,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_manifest": str(args.train_manifest.resolve()),
        "test_manifest": str(args.test_manifest.resolve()),
        "release_root": str(args.release_root.resolve()),
    }
    (args.output_root / "stage_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    for split, rows in (("train", train), ("test", test)):
        with (args.output_root / f"{split}_pairs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
