from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPOSITORY_ROOT / "runs"

EXPECTED_SPLIT_HASHES = {
    "run20_train.csv": "f765a1c63c020f60bc9fdd229078b84745ca1c1ba9ac8ea1dd7b249f15834f5a",
    "run20_test.csv": "c0b95db860ace22dc7f2633e30ced5d1aab9d05b7a835fb7a594b97c997d533d",
}

FORBIDDEN_PUBLIC_PATH_FRAGMENTS = (
    "/home/uturtle",
    "/mnt/nas_js",
    "\\Users\\uturtle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_expected_public_layout_exists() -> None:
    expected = (
        "README.md",
        "RUN_REGISTRY.csv",
        "SOURCE_COPY_MANIFEST.csv",
        "SHA256SUMS",
        "ERRATA_AND_PROVENANCE.md",
        "ADAPTATION_DIFF.md",
        "splits/dcase2026_seed13711/run20_train.csv",
        "splits/dcase2026_seed13711/run20_test.csv",
        "splits/dcase2026_seed13711/split_audit.json",
        "dcase_baseline_ae/source_snapshot",
        "dcase_baseline_ae/public_reproduction",
        "dcase_baseline_ae/results",
        "pretrained_encoder_knn/source_snapshot",
        "pretrained_encoder_knn/public_reproduction",
        "pretrained_encoder_knn/results",
        "pretrained_encoder_knn/external_models.json",
    )
    missing = [relative for relative in expected if not (RUNS_ROOT / relative).exists()]
    assert missing == []


def test_exact_split_hashes_and_session_disjointness() -> None:
    split_root = RUNS_ROOT / "splits/dcase2026_seed13711"
    rows_by_name: dict[str, list[dict[str, str]]] = {}
    for name, expected_hash in EXPECTED_SPLIT_HASHES.items():
        path = split_root / name
        assert sha256(path) == expected_hash
        with path.open(newline="", encoding="utf-8") as handle:
            rows_by_name[name] = list(csv.DictReader(handle))

    train = rows_by_name["run20_train.csv"]
    test = rows_by_name["run20_test.csv"]
    assert len(train) == 1000
    assert len(test) == 600
    assert {row["release_session_id"] for row in train}.isdisjoint(
        {row["release_session_id"] for row in test}
    )
    assert {row["clip_uid"] for row in train}.isdisjoint(
        {row["clip_uid"] for row in test}
    )
    assert {row["sampling_seed"] for row in train + test} == {"13711"}


def test_source_copy_manifest_matches_every_snapshot() -> None:
    manifest = RUNS_ROOT / "SOURCE_COPY_MANIFEST.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        destination = REPOSITORY_ROOT / row["repository_path"]
        assert destination.is_file()
        assert sha256(destination) == row["sha256"]
        assert int(row["bytes"]) == destination.stat().st_size


def test_sha256sums_cover_every_public_artifact() -> None:
    checksum_path = RUNS_ROOT / "SHA256SUMS"
    entries = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    expected_paths = {
        path.relative_to(RUNS_ROOT).as_posix()
        for path in RUNS_ROOT.rglob("*")
        if path.is_file()
        and path != checksum_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    assert set(entries) == expected_paths
    for relative, digest in entries.items():
        assert sha256(RUNS_ROOT / relative) == digest


def test_public_package_has_no_machine_local_paths_or_large_files() -> None:
    for path in RUNS_ROOT.rglob("*"):
        assert not path.is_symlink()
        if not path.is_file():
            continue
        assert path.stat().st_size < 10 * 1024 * 1024
        if path.suffix.lower() in {".csv", ".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8")
            assert not any(fragment in text for fragment in FORBIDDEN_PUBLIC_PATH_FRAGMENTS)


def test_generated_manifests_exclude_runtime_caches() -> None:
    checksum_text = (RUNS_ROOT / "SHA256SUMS").read_text(encoding="utf-8")
    assert "__pycache__" not in checksum_text
    assert ".pyc" not in checksum_text


def test_portable_entrypoints_support_help() -> None:
    scripts = (
        "dcase_baseline_ae/public_reproduction/scripts/build_split.py",
        "dcase_baseline_ae/public_reproduction/scripts/stage_split.py",
        "dcase_baseline_ae/public_reproduction/src/runner.py",
        "pretrained_encoder_knn/public_reproduction/scripts/stage_asd3dp_pairs.py",
        "pretrained_encoder_knn/public_reproduction/scripts/download_sslam.py",
        "pretrained_encoder_knn/public_reproduction/run_projection_select.py",
    )
    for relative in scripts:
        completed = subprocess.run(
            [sys.executable, str(RUNS_ROOT / relative), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()


def test_registry_uses_public_run_names_and_distinct_seeds() -> None:
    with (RUNS_ROOT / "RUN_REGISTRY.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_id"] for row in rows] == [
        "dcase_baseline_ae",
        "pretrained_encoder_knn",
    ]
    assert [row["display_name"] for row in rows] == [
        "DCASE baseline AE",
        "Pretrained encoder k-NN",
    ]
    assert all("internal_source" not in row for row in rows)
    assert all("Run 20" not in value and "Run 21" not in value for row in rows for value in row.values())
    assert all((RUNS_ROOT / row["paper_outputs"]).is_file() for row in rows)
    assert rows[0]["split_seed"] == "13711"
    assert rows[0]["model_seeds"] == "23711;23712;23713"
    assert rows[1]["split_seed"] == "13711"
    assert rows[1]["model_seeds"] == "deterministic_frozen_encoder"


def test_generated_manifests_are_current() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNS_ROOT / "tools/build_manifests.py"), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_reported_metrics_recompute_from_public_score_evidence() -> None:
    verifiers = (
        "dcase_baseline_ae/public_reproduction/scripts/verify_reported_metrics.py",
        "pretrained_encoder_knn/public_reproduction/scripts/verify_reported_metrics.py",
    )
    for relative in verifiers:
        completed = subprocess.run(
            [sys.executable, str(RUNS_ROOT / relative)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "validated" in completed.stdout.lower()
