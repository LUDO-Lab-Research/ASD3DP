#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


RUN_PACKAGE = Path(__file__).resolve().parents[2]
RESULTS = RUN_PACKAGE / "results"


def close(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=0.0, atol=1e-12))


def main() -> int:
    with (RESULTS / "run21_submitted_front_rear_table.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        table = list(csv.DictReader(handle))
    with (RESULTS / "post_submission_verification/run21_file_scores.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        score_rows = list(csv.DictReader(handle))

    for expected in table:
        rows = [row for row in score_rows if row["system"] == expected["system_id"]]
        labels = np.asarray(
            [row["benchmark_label"] == "anomaly" for row in rows], dtype=np.int64
        )
        scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        threshold = float(expected["train_normal_q95_threshold"])
        actual = {
            "auc": roc_auc_score(labels, scores),
            "pauc": roc_auc_score(labels, scores, max_fpr=0.1),
            "f1_train_normal_q95": f1_score(labels, scores > threshold),
        }
        if len(rows) != 600 or not all(
            close(actual[key], float(expected[key])) for key in actual
        ):
            raise RuntimeError(
                f"{expected['system_id']} metric mismatch: "
                f"actual={actual} expected={expected}"
            )
        print(f"validated {expected['system_id']}: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
