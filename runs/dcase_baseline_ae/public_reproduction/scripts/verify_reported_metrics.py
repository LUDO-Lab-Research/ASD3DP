#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


RUN_PACKAGE = Path(__file__).resolve().parents[2]
RESULTS = RUN_PACKAGE / "results"
SEEDS = (23711, 23712, 23713)


def close(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=0.0, atol=1e-12))


def main() -> int:
    for seed in SEEDS:
        seed_dir = RESULTS / f"seed_{seed}"
        with (seed_dir / "file_scores.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
        threshold = json.loads(
            (seed_dir / "gamma_threshold.json").read_text(encoding="utf-8")
        )["threshold"]
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        actual = {
            "auc": roc_auc_score(labels, scores),
            "pauc": roc_auc_score(labels, scores, max_fpr=0.1),
            "f1": f1_score(labels, scores > threshold),
        }
        if len(rows) != 600 or not all(close(actual[key], expected[key]) for key in actual):
            raise RuntimeError(
                f"seed {seed} metric mismatch: actual={actual} expected={expected}"
            )
        print(f"validated seed {seed}: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
