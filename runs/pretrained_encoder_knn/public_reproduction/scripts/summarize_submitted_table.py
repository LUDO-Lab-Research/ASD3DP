#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))

from run_projection_select import (  # noqa: E402
    load_projection_config,
    machine_meta,
    system_scores,
)


config = load_projection_config(RUN / "config/run21_asd3dp.yaml")
meta = machine_meta(config.train_split, config.eval_split, "ASD3DP")
labels = np.asarray(meta["test_labels"], dtype=np.int64)
rows = []
for system in config.systems:
    scores, train_scores = system_scores(system, config, "ASD3DP")
    threshold = float(np.quantile(train_scores, 0.95))
    rows.append(
        {
            "system_id": system.system_id,
            "view": {
                "front_only": "Front-only",
                "near_only": "Rear-only",
                "residual_view": "Rear - 0.5 front",
            }[system.system_id],
            "auc": roc_auc_score(labels, scores),
            "pauc": roc_auc_score(labels, scores, max_fpr=0.1),
            "f1_train_normal_q95": f1_score(labels, scores > threshold),
            "train_normal_q95_threshold": threshold,
        }
    )

path = RUN / "out/run21_submitted_front_rear_table.csv"
with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(path)
