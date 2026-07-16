#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path


RUNS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = RUNS_ROOT.parent
SOURCE_MANIFEST = RUNS_ROOT / "SOURCE_COPY_MANIFEST.csv"
CHECKSUM_MANIFEST = RUNS_ROOT / "SHA256SUMS"

EXACT_NON_SNAPSHOT_FILES = (
    "splits/dcase2026_seed13711/run20_train.csv",
    "splits/dcase2026_seed13711/run20_test.csv",
    "dcase_baseline_ae/results/run20_seed_metrics.csv",
    "dcase_baseline_ae/results/run20_summary_metrics.csv",
    "dcase_baseline_ae/results/run20_fault_seed_metrics.csv",
    "dcase_baseline_ae/results/run20_fault_summary_metrics.csv",
    "dcase_baseline_ae/results/seed_23711/metrics.json",
    "dcase_baseline_ae/results/seed_23711/gamma_threshold.json",
    "dcase_baseline_ae/results/seed_23712/metrics.json",
    "dcase_baseline_ae/results/seed_23712/gamma_threshold.json",
    "dcase_baseline_ae/results/seed_23713/metrics.json",
    "dcase_baseline_ae/results/seed_23713/gamma_threshold.json",
    "pretrained_encoder_knn/results/run21_submitted_front_rear_table.csv",
    "pretrained_encoder_knn/results/run21_label_free_f1.csv",
    "pretrained_encoder_knn/results/rear_front/result.csv",
    "pretrained_encoder_knn/results/rear_front/run21_fault_metrics.csv",
    "pretrained_encoder_knn/results/rear_front/run21_file_scores.csv",
    "pretrained_encoder_knn/results/rear_front/resolved_config.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(repository_relative: str) -> str:
    if repository_relative.startswith("runs/dcase_baseline_ae/source_snapshot/shared/"):
        suffix = repository_relative.split("/shared/", 1)[1]
        return f"runs/03_gcode_aligned_no_audio_threshold/src/{suffix}"
    if repository_relative.startswith("runs/dcase_baseline_ae/source_snapshot/"):
        suffix = repository_relative.split("/source_snapshot/", 1)[1]
        return f"runs/20_single_dcase_ae_baseline/{suffix}"
    if repository_relative.startswith("runs/pretrained_encoder_knn/source_snapshot/"):
        suffix = repository_relative.split("/source_snapshot/", 1)[1]
        return f"runs/21_genrep_baseline/{suffix}"
    if repository_relative.startswith("runs/splits/"):
        name = Path(repository_relative).name
        return f"runs/20_single_dcase_ae_baseline/splits/{name}"
    if repository_relative.startswith("runs/dcase_baseline_ae/results/"):
        suffix = repository_relative.split("/results/", 1)[1]
        return f"runs/20_single_dcase_ae_baseline/outputs/{suffix}"
    if repository_relative.startswith("runs/pretrained_encoder_knn/results/"):
        suffix = repository_relative.split("/results/", 1)[1]
        if suffix.startswith("rear_front/"):
            suffix = f"run21_scores/{suffix.split('/', 1)[1]}"
        return f"runs/21_genrep_baseline/out/{suffix}"
    raise ValueError(f"no source mapping for {repository_relative}")


def source_manifest_text() -> str:
    snapshots = sorted(
        path
        for path in RUNS_ROOT.rglob("*")
        if path.is_file() and "source_snapshot" in path.parts
    )
    exact_files = [RUNS_ROOT / relative for relative in EXACT_NON_SNAPSHOT_FILES]
    rows = []
    for path in snapshots + exact_files:
        repository_path = path.relative_to(REPOSITORY_ROOT).as_posix()
        rows.append(
            {
                "source_id": source_id(repository_path),
                "repository_path": repository_path,
                "bytes": str(path.stat().st_size),
                "sha256": sha256(path),
                "copy_status": "byte_identical",
            }
        )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=(
            "source_id",
            "repository_path",
            "bytes",
            "sha256",
            "copy_status",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: row["repository_path"]))
    return output.getvalue()


def checksum_text() -> str:
    paths = sorted(
        path
        for path in RUNS_ROOT.rglob("*")
        if path.is_file()
        and path != CHECKSUM_MANIFEST
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    return "".join(
        f"{sha256(path)}  {path.relative_to(RUNS_ROOT).as_posix()}\n"
        for path in paths
    )


def write_or_check(path: Path, expected: str, check: bool) -> None:
    if check:
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            raise RuntimeError(f"generated manifest is stale: {path}")
    else:
        path.write_text(expected, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_or_check(SOURCE_MANIFEST, source_manifest_text(), args.check)
    write_or_check(CHECKSUM_MANIFEST, checksum_text(), args.check)
    print(f"{'checked' if args.check else 'wrote'} {SOURCE_MANIFEST}")
    print(f"{'checked' if args.check else 'wrote'} {CHECKSUM_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
