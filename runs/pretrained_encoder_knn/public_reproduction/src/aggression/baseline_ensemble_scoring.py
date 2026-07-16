"""Score-level late fusion for Baseline encoder ensembles.

This script averages clip-wise anomaly scores already produced by GenRep-style
encoder runs. It does not fuse features and does not use labels, domains, or
test-set score statistics.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
)


DEFAULT_SCORE_GLOB = "anomaly_score_*_section_00_test.csv"


MACHINE_RE = re.compile(r"^anomaly_score_(.+)_section_00_test\.csv$")


def eval_score(
    gt_list: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float, float]:
    """Match the Baseline runner's AUC, pAUC, and best-F1 metric."""
    gt_list = np.asarray(gt_list)
    scores = np.asarray(scores)
    auc = roc_auc_score(gt_list, scores)
    pauc = roc_auc_score(gt_list, scores, max_fpr=0.1)
    precision, recall, _ = precision_recall_curve(gt_list, scores)
    f1_scores = (
        (2 * precision * recall)
        / (precision + recall + np.finfo(float).eps)
    )
    return auc, pauc, float(f1_scores[np.argmax(f1_scores)])


def parse_machine_name(score_name: str) -> str:
    """Parse machine name from official-style anomaly score filename."""
    match = MACHINE_RE.match(score_name)
    if match is None:
        raise ValueError(f"Cannot parse machine name from {score_name}.")
    return match.group(1)


def parse_dev_valid_label_domain(
    clips: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse public dev valid label and source/target domain from filenames."""
    gt_list = []
    source_list = []
    for clip in clips:
        clip_name = Path(str(clip)).name
        parts = clip_name.split("_")
        if "normal" in parts:
            gt_list.append(0)
        elif "anomaly" in parts:
            gt_list.append(1)
        else:
            raise ValueError(f"Cannot parse normal/anomaly label: {clip}")

        if "source" in parts:
            source_list.append(True)
        elif "target" in parts:
            source_list.append(False)
        else:
            raise ValueError(f"Cannot parse source/target domain: {clip}")

    return np.asarray(gt_list), np.asarray(source_list)


def evaluate_score_dir(score_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate averaged raw scores with the Baseline runner metric split."""
    rows = []
    for score_path in sorted(score_dir.glob(DEFAULT_SCORE_GLOB)):
        df_score = read_score_file(score_path)
        gt_list, source_list = parse_dev_valid_label_domain(df_score["clip"])
        scores = df_score["score"].to_numpy()

        source_mask = source_list | (gt_list != 0)
        target_mask = (~source_list) | (gt_list != 0)
        gt_source = gt_list[source_mask]
        gt_target = gt_list[target_mask]
        score_source = scores[source_mask]
        score_target = scores[target_mask]

        auc_all, pauc_all, f1_all = eval_score(gt_list, scores)
        auc_source, pauc_source, f1_source = eval_score(
            gt_source,
            score_source,
        )
        auc_target, pauc_target, f1_target = eval_score(
            gt_target,
            score_target,
        )
        official_score = hmean([auc_source, auc_target, pauc_all])

        rows.append(
            {
                "machine": parse_machine_name(score_path.name),
                "auc": auc_all,
                "pauc": pauc_all,
                "f1": f1_all,
                "auc_source": auc_source,
                "auc_target": auc_target,
                "official_score": official_score,
                "pauc_source": pauc_source,
                "pauc_target": pauc_target,
                "f1_source": f1_source,
                "f1_target": f1_target,
                "score_file": score_path.name,
            }
        )

    if not rows:
        raise ValueError(f"No score files found for evaluation in {score_dir}.")

    df_machine = pd.DataFrame(rows)
    metric_columns = [
        "auc",
        "pauc",
        "f1",
        "auc_source",
        "auc_target",
        "official_score",
        "pauc_source",
        "pauc_target",
        "f1_source",
        "f1_target",
    ]
    df_final = pd.DataFrame(df_machine[metric_columns].agg(hmean)).T
    return df_machine, df_final


def read_score_file(score_path: Path) -> pd.DataFrame:
    """Read one official-style anomaly score CSV."""
    df_score = pd.read_csv(score_path, header=None)
    if df_score.shape[1] != 2:
        raise ValueError(
            f"{score_path} must have exactly 2 columns: clip path and score."
        )

    df_score.columns = ["clip", "score"]
    if df_score["clip"].duplicated().any():
        duplicated = df_score.loc[
            df_score["clip"].duplicated(), "clip"
        ].iloc[0]
        raise ValueError(f"{score_path} has duplicated clip id: {duplicated}")

    df_score["score"] = pd.to_numeric(df_score["score"], errors="raise")
    if df_score["score"].isna().any():
        raise ValueError(f"{score_path} contains NaN scores.")

    return df_score


def collect_score_files(score_dir: Path, score_glob: str) -> dict[str, Path]:
    """Collect score files in a directory by basename."""
    score_files = sorted(score_dir.glob(score_glob))
    if not score_files:
        raise ValueError(
            f"No score files found in {score_dir} with glob {score_glob}."
        )

    by_name = {score_file.name: score_file for score_file in score_files}
    if len(by_name) != len(score_files):
        raise ValueError(f"{score_dir} contains duplicate score basenames.")

    return by_name


def validate_score_file_sets(
    files_by_dir: list[dict[str, Path]],
    score_dirs: list[Path],
) -> list[str]:
    """Require every ensemble member to expose the same score file names."""
    expected_names = set(files_by_dir[0])
    for score_dir, score_files in zip(score_dirs[1:], files_by_dir[1:]):
        current_names = set(score_files)
        missing = sorted(expected_names - current_names)
        extra = sorted(current_names - expected_names)
        if missing or extra:
            raise ValueError(
                "Score file set mismatch for "
                f"{score_dir}. missing={missing}, extra={extra}"
            )

    return sorted(expected_names)


def average_score_file(score_paths: list[Path]) -> pd.DataFrame:
    """Average one machine score file across ensemble members."""
    base_score = read_score_file(score_paths[0])
    base_clips = set(base_score["clip"])
    merged = base_score.rename(columns={"score": "score_0"})
    for index, score_path in enumerate(score_paths[1:], start=1):
        current_score = read_score_file(score_path)
        current_clips = set(current_score["clip"])
        missing = sorted(base_clips - current_clips)
        extra = sorted(current_clips - base_clips)
        if missing or extra:
            raise ValueError(
                f"Clip ids do not match for {score_path}. "
                f"missing={missing}, extra={extra}"
            )

        current_score = current_score.rename(
            columns={"score": f"score_{index}"}
        )
        merged = merged.merge(
            current_score,
            on="clip",
            how="left",
            validate="1:1",
        )

    score_columns = [column for column in merged.columns if column != "clip"]
    df_out = pd.DataFrame()
    df_out["clip"] = merged["clip"]
    df_out["score"] = merged[score_columns].mean(axis=1)
    return df_out


def write_ensemble_scores(
    score_dirs: list[Path],
    output_dir: Path,
    score_glob: str = DEFAULT_SCORE_GLOB,
    decision_threshold: float | None = None,
) -> pd.DataFrame:
    """Write averaged anomaly score files and a compact manifest."""
    if len(score_dirs) < 2:
        raise ValueError("At least two --score_dir values are required.")

    files_by_dir = [
        collect_score_files(score_dir, score_glob) for score_dir in score_dirs
    ]
    score_names = validate_score_file_sets(files_by_dir, score_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for score_name in score_names:
        score_paths = [score_files[score_name] for score_files in files_by_dir]
        df_score = average_score_file(score_paths)
        output_path = output_dir / score_name
        df_score.to_csv(output_path, header=False, index=False)

        decision_path = None
        if decision_threshold is not None:
            decision_name = score_name.replace(
                "anomaly_score_", "decision_result_", 1
            )
            decision_path = output_dir / decision_name
            df_decision = pd.DataFrame()
            df_decision["clip"] = df_score["clip"]
            df_decision["decision"] = (
                df_score["score"] > decision_threshold
            ).astype(int)
            df_decision.to_csv(decision_path, header=False, index=False)

        manifest_rows.append(
            {
                "score_file": score_name,
                "output_path": str(output_path),
                "decision_path": "" if decision_path is None else str(
                    decision_path
                ),
                "n_clips": len(df_score),
                "n_members": len(score_dirs),
                "source_dirs": ";".join(str(path) for path in score_dirs),
            }
        )

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(output_dir / "ensemble_manifest.csv", index=False)
    return df_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average GenRep-style anomaly score CSV files."
    )
    parser.add_argument(
        "--score_dir",
        action="append",
        required=True,
        help="Directory containing anomaly_score_*_section_00_test.csv files. "
        "Pass once per encoder/layer score source.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where averaged score CSV files will be written.",
    )
    parser.add_argument(
        "--score_glob",
        default=DEFAULT_SCORE_GLOB,
        help=f"Score filename glob. Default: {DEFAULT_SCORE_GLOB}",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=None,
        help="Optional fixed score threshold for decision_result files. "
        "By default decisions are not generated.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate averaged dev-valid score files with raw Baseline metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_dirs = [Path(score_dir) for score_dir in args.score_dir]
    output_dir = Path(args.output_dir)

    df_manifest = write_ensemble_scores(
        score_dirs=score_dirs,
        output_dir=output_dir,
        score_glob=args.score_glob,
        decision_threshold=args.decision_threshold,
    )
    print(df_manifest)

    if args.evaluate:
        df_machine, df_final = evaluate_score_dir(output_dir)
        df_machine.to_csv(output_dir / "machine_scores.csv", index=False)
        df_final.to_csv(output_dir / "final_score.csv", index=False)
        print("ENSEMBLE MACHINE SCORES")
        print(df_machine)
        print("ENSEMBLE FINAL SCORE")
        print(df_final)

    print(f"Save ensemble scores to {output_dir}")


if __name__ == "__main__":
    main()
