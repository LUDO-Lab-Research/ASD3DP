#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def stage(manifest: Path, release_root: Path, output: Path, split: str) -> int:
    target = output / split
    target.mkdir(parents=True, exist_ok=True)
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        label = row["benchmark_label"]
        source = release_root / row["clip_audio_file"]
        name = f"section_00_{split}_{label}_{index:04d}_{row['clip_uid']}.wav"
        destination = target / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.symlink(source, destination)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    train_count = stage(args.train_manifest, args.release_root, args.output, "train")
    test_count = stage(args.test_manifest, args.release_root, args.output, "test")
    if (train_count, test_count) != (1000, 600):
        raise RuntimeError((train_count, test_count))
    print(f"train={train_count} test={test_count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
