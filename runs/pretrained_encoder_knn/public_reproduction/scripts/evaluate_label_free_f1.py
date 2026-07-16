#!/usr/bin/env python3
"""Evaluate Run 21 with the configured train-normal quantile threshold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_projection_select as runner


def cached_scores(
    system: runner.SystemSpec,
    config: runner.ProjectionConfig,
    cache_machine: str,
) -> tuple[np.ndarray, np.ndarray]:
    test_parts = []
    train_parts = []
    for spec in system.encoders:
        train_views = runner.make_views(
            runner.load_pair_features(
                spec, config.train_split, cache_machine, config.cache_root_template
            ),
            spec.layer,
            config.residual_alpha,
        )
        test_views = runner.make_views(
            runner.load_pair_features(
                spec, config.eval_split, cache_machine, config.cache_root_template
            ),
            spec.layer,
            config.residual_alpha,
        )
        view = "near" if system.mode == "near" else "residual"
        bank = train_views[view]
        test_parts.append(runner.nearest_scores(test_views[view], bank))
        train_parts.append(runner.loo_scores(bank, bank))
    return np.mean(np.stack(test_parts), axis=0), np.mean(np.stack(train_parts), axis=0)


def evaluate(config_path: Path, cache_machine: str | None) -> list[dict[str, object]]:
    config = runner.load_projection_config(config_path)
    train_data = runner.get_dcase2026(config.train_split)["train"]
    machines = sorted(np.unique(train_data["machine_names"]).tolist())
    rows: list[dict[str, object]] = []
    if len(machines) != 1:
        raise ValueError(f"expected one metadata machine, found {machines}")
    metadata_machine = machines[0]
    resolved_cache_machine = cache_machine or metadata_machine
    for system in config.systems:
        meta = runner.machine_meta(
            config.train_split, config.eval_split, metadata_machine
        )
        test_scores, train_scores = cached_scores(
            system, config, resolved_cache_machine
        )
        labels = runner.collapse_if_needed(
            np.asarray(meta["test_labels"]), len(test_scores)
        ).astype(int)
        threshold = float(np.quantile(train_scores, config.decision_q))
        predictions = (test_scores > threshold).astype(int)
        rows.append(
            {
                "config": str(config_path.relative_to(ROOT)),
                "cache_machine": resolved_cache_machine,
                "system_id": system.system_id,
                "system": system.label,
                "decision_quantile": config.decision_q,
                "threshold": threshold,
                "train_score_count": len(train_scores),
                "test_score_count": len(test_scores),
                "predicted_anomaly_count": int(predictions.sum()),
                "precision": precision_score(labels, predictions, zero_division=0),
                "recall": recall_score(labels, predictions, zero_division=0),
                "f1": f1_score(labels, predictions, zero_division=0),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-machine")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    resolved = args.config if args.config.is_absolute() else ROOT / args.config
    rows = evaluate(resolved, args.cache_machine)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        output,
        mode="a" if args.append else "w",
        header=not args.append,
        index=False,
    )
    print(frame.to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
