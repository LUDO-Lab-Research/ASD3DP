"""Layer-wise anomaly score fusion for score aggression ablations.

This module fuses already-computed clip-wise anomaly scores across layers.
It does not fuse encoder features and does not use labels, domains, or
test-time score statistics for the fusion itself.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggression.baseline_ensemble_scoring import (
    DEFAULT_SCORE_GLOB,
    evaluate_score_dir,
    read_score_file,
)


LAYER_DIR_RE = re.compile(r"^layer_(\d+)$")

DEFAULT_LAYER_SETS: dict[str, tuple[int, ...]] = {
    "all": tuple(range(1, 13)),
    "low": tuple(range(1, 7)),
    "mid": tuple(range(3, 10)),
    "high": tuple(range(6, 13)),
    "low_high": tuple(range(1, 4)) + tuple(range(9, 13)),
    "low_mid": tuple(range(1, 10)),
    "mid_high": tuple(range(3, 13)),
}

DEFAULT_FUSION_METHODS = ("mean", "min", "max")


def parse_layer_csv(value: str) -> tuple[int, ...]:
    """Parse `1,2,3` or `1-3,9-12` into a sorted unique layer tuple."""
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid layer range: {part}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))

    if not layers:
        raise ValueError("Layer list must not be empty.")
    if any(layer <= 0 for layer in layers):
        raise ValueError(f"Layer ids must be positive: {value}")

    return tuple(sorted(set(layers)))


def parse_custom_layer_sets(values: list[str] | None) -> dict[str, tuple[int, ...]]:
    """Parse repeated `name=1-3,9-12` layer-set arguments."""
    parsed: dict[str, tuple[int, ...]] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                "--layer_set must use name=layers format, "
                f"for example low_high=1-3,9-12; got {value}"
            )
        name, layers_text = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Layer-set name is empty: {value}")
        if name in parsed:
            raise ValueError(f"Duplicated layer-set name: {name}")
        parsed[name] = parse_layer_csv(layers_text)
    return parsed


def discover_layer_ids(score_root: Path) -> tuple[int, ...]:
    """Find available `layer_XX` directories under a score root."""
    layers = []
    for child in score_root.iterdir():
        if not child.is_dir():
            continue
        match = LAYER_DIR_RE.match(child.name)
        if match is not None:
            layers.append(int(match.group(1)))

    if not layers:
        raise ValueError(f"No layer_XX directories found in {score_root}.")
    return tuple(sorted(layers))


def layer_dir(score_root: Path, layer: int) -> Path:
    """Return the canonical layer directory path."""
    return score_root / f"layer_{layer:02d}"


def validate_layers(score_root: Path, layers: tuple[int, ...]) -> None:
    """Require every selected layer directory to exist."""
    missing = [layer for layer in layers if not layer_dir(score_root, layer).is_dir()]
    if missing:
        available = discover_layer_ids(score_root)
        raise ValueError(
            f"Missing layer directories for {score_root}: {missing}. "
            f"Available layers: {available}"
        )


def collect_score_names(
    score_root: Path,
    layers: tuple[int, ...],
    score_glob: str,
) -> list[str]:
    """Require selected layers to expose the same score file names."""
    expected_names: set[str] | None = None
    for layer in layers:
        current_names = {
            score_path.name
            for score_path in layer_dir(score_root, layer).glob(score_glob)
        }
        if not current_names:
            raise ValueError(
                f"No score files found in {layer_dir(score_root, layer)} "
                f"with glob {score_glob}."
            )
        if expected_names is None:
            expected_names = current_names
            continue

        missing = sorted(expected_names - current_names)
        extra = sorted(current_names - expected_names)
        if missing or extra:
            raise ValueError(
                f"Score file mismatch at layer {layer}. "
                f"missing={missing}, extra={extra}"
            )

    return sorted(expected_names or set())


def fuse_scores(values: np.ndarray, method: str) -> np.ndarray:
    """Fuse a `[n_clips, n_layers]` score matrix."""
    if method == "mean":
        return values.mean(axis=1)
    if method == "min":
        return values.min(axis=1)
    if method == "max":
        return values.max(axis=1)
    raise ValueError(f"Unsupported fusion method: {method}")


def fuse_one_score_file(
    score_root: Path,
    layers: tuple[int, ...],
    score_name: str,
    method: str,
) -> pd.DataFrame:
    """Fuse one machine score file across selected layers."""
    base = read_score_file(layer_dir(score_root, layers[0]) / score_name)
    merged = base.rename(columns={"score": f"layer_{layers[0]:02d}"})

    for layer in layers[1:]:
        current = read_score_file(layer_dir(score_root, layer) / score_name)
        current = current.rename(columns={"score": f"layer_{layer:02d}"})
        merged = merged.merge(current, on="clip", how="left", validate="1:1")

    layer_columns = [f"layer_{layer:02d}" for layer in layers]
    if merged[layer_columns].isna().any().any():
        raise ValueError(f"Merged score contains NaN values: {score_name}")

    df_out = pd.DataFrame()
    df_out["clip"] = merged["clip"]
    df_out["score"] = fuse_scores(merged[layer_columns].to_numpy(), method)
    return df_out


def write_layer_fusion_scores(
    score_root: Path,
    output_root: Path,
    layer_sets: dict[str, tuple[int, ...]],
    methods: tuple[str, ...] = DEFAULT_FUSION_METHODS,
    score_glob: str = DEFAULT_SCORE_GLOB,
) -> pd.DataFrame:
    """Write fused score files for every layer-set and fusion method."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for layer_set_name, layers in layer_sets.items():
        validate_layers(score_root, layers)
        score_names = collect_score_names(score_root, layers, score_glob)

        for method in methods:
            if method not in DEFAULT_FUSION_METHODS:
                raise ValueError(
                    f"Unsupported method {method}; "
                    f"valid methods are {DEFAULT_FUSION_METHODS}."
                )

            output_dir = output_root / layer_set_name / method
            output_dir.mkdir(parents=True, exist_ok=True)
            for score_name in score_names:
                df_score = fuse_one_score_file(
                    score_root=score_root,
                    layers=layers,
                    score_name=score_name,
                    method=method,
                )
                df_score.to_csv(output_dir / score_name, header=False, index=False)

            manifest_rows.append(
                {
                    "layer_set": layer_set_name,
                    "method": method,
                    "layers": ",".join(str(layer) for layer in layers),
                    "n_layers": len(layers),
                    "n_score_files": len(score_names),
                    "output_dir": str(output_dir),
                }
            )

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(output_root / "layer_score_fusion_manifest.csv", index=False)
    return df_manifest


def evaluate_layer_fusions(
    manifest: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every fused score directory using raw Baseline metrics."""
    machine_rows = []
    final_rows = []
    for row in manifest.to_dict("records"):
        score_dir = Path(row["output_dir"])
        df_machine, df_final = evaluate_score_dir(score_dir)

        for column in ("layer_set", "method", "layers", "n_layers"):
            df_machine[column] = row[column]
            df_final[column] = row[column]

        df_machine.to_csv(score_dir / "machine_scores.csv", index=False)
        df_final.to_csv(score_dir / "final_score.csv", index=False)
        machine_rows.append(df_machine)
        final_rows.append(df_final)

    df_machine_all = pd.concat(machine_rows, ignore_index=True)
    df_final_all = pd.concat(final_rows, ignore_index=True)

    preferred_columns = ["layer_set", "method", "layers", "n_layers"]
    metric_columns = [
        column for column in df_final_all.columns if column not in preferred_columns
    ]
    df_machine_all = df_machine_all[
        ["layer_set", "method", "layers", "n_layers", "machine"]
        + [
            column
            for column in df_machine_all.columns
            if column
            not in {"layer_set", "method", "layers", "n_layers", "machine"}
        ]
    ]
    df_final_all = df_final_all[preferred_columns + metric_columns]

    df_machine_all.to_csv(output_root / "machine_scores_all.csv", index=False)
    df_final_all.to_csv(output_root / "final_scores_all.csv", index=False)
    return df_machine_all, df_final_all


def default_layer_sets_for(score_root: Path) -> dict[str, tuple[int, ...]]:
    """Use default 12-layer sets, or `all` only for non-12-layer encoders."""
    available_layers = discover_layer_ids(score_root)
    if set(range(1, 13)).issubset(available_layers):
        return DEFAULT_LAYER_SETS
    return {"all": available_layers}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse GenRep-style anomaly scores across representation layers."
    )
    parser.add_argument(
        "--score_root",
        required=True,
        help="Directory containing layer_XX/anomaly_score_*_section_00_test.csv.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Directory where fused score files and summaries are written.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=DEFAULT_FUSION_METHODS,
        help="Fusion method. Repeatable. Default: mean, min, max.",
    )
    parser.add_argument(
        "--layer_set",
        action="append",
        help="Custom layer set in name=layers format, e.g. high=6-12. "
        "Repeatable. If omitted, default 12-layer sets are used.",
    )
    parser.add_argument(
        "--score_glob",
        default=DEFAULT_SCORE_GLOB,
        help=f"Score filename glob. Default: {DEFAULT_SCORE_GLOB}",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate fused dev-valid score files with raw Baseline metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_root = Path(args.score_root)
    output_root = Path(args.output_root)
    methods = tuple(args.method or DEFAULT_FUSION_METHODS)
    layer_sets = (
        parse_custom_layer_sets(args.layer_set)
        if args.layer_set
        else default_layer_sets_for(score_root)
    )

    manifest = write_layer_fusion_scores(
        score_root=score_root,
        output_root=output_root,
        layer_sets=layer_sets,
        methods=methods,
        score_glob=args.score_glob,
    )
    print("LAYER SCORE FUSION MANIFEST")
    print(manifest)

    if args.evaluate:
        _, df_final = evaluate_layer_fusions(manifest, output_root)
        print("LAYER SCORE FUSION FINAL SCORES")
        print(df_final)

    print(f"Save layer score fusion outputs to {output_root}")


if __name__ == "__main__":
    main()
