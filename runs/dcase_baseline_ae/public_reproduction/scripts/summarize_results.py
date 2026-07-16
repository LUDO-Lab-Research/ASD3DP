#!/usr/bin/env python3
import csv
import json
import statistics
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
OUT = RUN.parent / "results"
SEEDS = (23711, 23712, 23713)
FIELDS = ("auc", "pauc", "f1", "gamma_threshold")

rows = []
for seed in SEEDS:
    metrics = json.loads((OUT / f"seed_{seed}" / "metrics.json").read_text())
    gamma = json.loads((OUT / f"seed_{seed}" / "gamma_threshold.json").read_text())
    rows.append(
        {
            "seed": seed,
            "auc": metrics["auc"],
            "pauc": metrics["pauc"],
            "f1": metrics["f1"],
            "gamma_threshold": gamma["threshold"],
            "n_normal": metrics["n_normal"],
            "n_anomaly": metrics["n_anomaly"],
        }
    )

seed_path = OUT / "run20_seed_metrics.csv"
with seed_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = []
for field in FIELDS:
    values = [float(row[field]) for row in rows]
    summary.append(
        {
            "metric": field,
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "n_seeds": len(values),
        }
    )

summary_path = OUT / "run20_summary_metrics.csv"
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print(seed_path)
print(summary_path)
