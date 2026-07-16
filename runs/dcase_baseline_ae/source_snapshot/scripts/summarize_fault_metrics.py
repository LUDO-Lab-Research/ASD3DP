#!/usr/bin/env python3
import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


RUN = Path(__file__).resolve().parents[1]
OUTPUTS = RUN / "outputs"
SEEDS = (23711, 23712, 23713)
FAULTS = (
    "belt_tension_abnormal",
    "extruder_cogging_or_no_extrusion",
    "toolhead_collision",
)
INDEX_RE = re.compile(r"_test_(?:normal|anomaly)_(\d{4})_")

with (RUN / "splits" / "run20_test.csv").open(newline="") as handle:
    manifest = list(csv.DictReader(handle))

seed_rows = []
for seed in SEEDS:
    with (OUTPUTS / f"seed_{seed}" / "file_scores.csv").open(newline="") as handle:
        scores = list(csv.DictReader(handle))
    threshold = json.loads(
        (OUTPUTS / f"seed_{seed}" / "gamma_threshold.json").read_text()
    )["threshold"]
    joined = []
    for score_row in scores:
        match = INDEX_RE.search(score_row["basename"])
        if not match:
            raise ValueError(f"cannot recover manifest index: {score_row['basename']}")
        manifest_row = manifest[int(match.group(1))]
        joined.append(
            {
                "label": int(score_row["label"]),
                "score": float(score_row["score"]),
                "fault_mode": manifest_row["fault_mode"],
            }
        )
    normals = [row for row in joined if row["label"] == 0]
    if len(normals) != 300:
        raise ValueError(f"expected 300 normal rows, found {len(normals)}")
    for fault in FAULTS:
        anomalies = [row for row in joined if row["fault_mode"] == fault]
        if len(anomalies) != 100:
            raise ValueError(f"expected 100 {fault} rows, found {len(anomalies)}")
        subset = normals + anomalies
        labels = np.asarray([row["label"] for row in subset], dtype=np.int64)
        values = np.asarray([row["score"] for row in subset], dtype=np.float64)
        seed_rows.append(
            {
                "seed": seed,
                "fault_mode": fault,
                "n_normal": len(normals),
                "n_anomaly": len(anomalies),
                "auc": roc_auc_score(labels, values),
                "pauc": roc_auc_score(labels, values, max_fpr=0.1),
                "f1": f1_score(labels, values > threshold),
                "gamma_threshold": threshold,
            }
        )

seed_path = OUTPUTS / "run20_fault_seed_metrics.csv"
with seed_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
    writer.writeheader()
    writer.writerows(seed_rows)

summary_rows = []
for fault in FAULTS:
    selected = [row for row in seed_rows if row["fault_mode"] == fault]
    row = {"fault_mode": fault, "n_normal": 300, "n_anomaly": 100, "n_seeds": 3}
    for metric in ("auc", "pauc", "f1"):
        values = [float(item[metric]) for item in selected]
        row[f"{metric}_mean"] = statistics.mean(values)
        row[f"{metric}_sample_sd"] = statistics.stdev(values)
    summary_rows.append(row)

summary_path = OUTPUTS / "run20_fault_summary_metrics.csv"
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    writer.writerows(summary_rows)

print(seed_path)
print(summary_path)
