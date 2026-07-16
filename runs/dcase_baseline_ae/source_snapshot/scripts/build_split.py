#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


CONTEXTS = (
    "print_slow_loaded",
    "print_fast_loaded",
    "motion_slow_no_filament",
    "motion_fast_no_filament",
)
CONTEXT_QUOTA = 250
SAMPLING_SEED = 13711
HELD_OUT_NORMAL_SESSIONS = {
    "print_slow_loaded": ("asd3dp_s0012",),
    "print_fast_loaded": ("asd3dp_s0015",),
    "motion_slow_no_filament": (),
    "motion_fast_no_filament": ("asd3dp_s0019",),
}
TRAIN_ONLY_NORMAL_SESSIONS = ("asd3dp_s0016",)
TEST_FAULTS = (
    "belt_tension_abnormal",
    "extruder_cogging_or_no_extrusion",
    "toolhead_collision",
)


def equal_quotas(session_ids: list[str], total: int) -> dict[str, int]:
    ordered = sorted(session_ids)
    if not ordered:
        raise ValueError("cannot allocate a quota without sessions")
    base, remainder = divmod(total, len(ordered))
    return {
        session_id: base + (index < remainder)
        for index, session_id in enumerate(ordered)
    }


def capacity_aware_quotas(capacities: dict[str, int], total: int) -> dict[str, int]:
    if sum(capacities.values()) < total:
        raise ValueError(f"pool capacity {sum(capacities.values())} is below quota {total}")
    quotas = {session_id: 0 for session_id in sorted(capacities)}
    remaining = total
    while remaining:
        eligible = [
            session_id
            for session_id in sorted(quotas)
            if quotas[session_id] < capacities[session_id]
        ]
        if not eligible:
            break
        minimum = min(quotas[session_id] for session_id in eligible)
        for session_id in eligible:
            if quotas[session_id] == minimum:
                quotas[session_id] += 1
                remaining -= 1
                if remaining == 0:
                    break
    if remaining:
        raise ValueError(f"failed to allocate {remaining} rows")
    return quotas


def sample_timeline(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: int(row["clip_index"]))
    if len(ordered) < count:
        raise ValueError(f"normal pool has {len(ordered)} rows but requires {count}")
    rng = random.Random(seed)
    selected = []
    for bin_index in range(count):
        start = bin_index * len(ordered) // count
        end = (bin_index + 1) * len(ordered) // count
        selected.append(ordered[rng.randrange(start, end)])
    return selected


def sample_training_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    by_context_session: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        context = row["protocol_axis"]
        if context in CONTEXTS:
            by_context_session[context][row["release_session_id"]].append(row)

    selected: list[dict[str, str]] = []
    for context_index, context in enumerate(CONTEXTS):
        sessions = by_context_session[context]
        quotas = equal_quotas(list(sessions), CONTEXT_QUOTA)
        for session_index, session_id in enumerate(sorted(sessions)):
            session_seed = seed + context_index * 10_000 + session_index
            selected.extend(sample_timeline(sessions[session_id], quotas[session_id], session_seed))
    return selected


def sample_balanced_sessions(
    rows: list[dict[str, str]], count: int, seed: int
) -> list[dict[str, str]]:
    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_session[row["release_session_id"]].append(row)
    quotas = capacity_aware_quotas(
        {session_id: len(session_rows) for session_id, session_rows in by_session.items()},
        count,
    )
    selected = []
    for index, session_id in enumerate(sorted(by_session)):
        selected.extend(
            sample_timeline(by_session[session_id], quotas[session_id], seed + index)
        )
    return selected


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio_path(release_root: Path, row: dict[str, str]) -> Path:
    return release_root / row["clip_audio_file"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SAMPLING_SEED)
    args = parser.parse_args()

    fields, rows = read_csv(args.manifest)
    front = [row for row in rows if row["channel"] == "ch2"]
    normal = [row for row in front if row["benchmark_label"] == "normal"]
    anomaly = [row for row in front if row["benchmark_label"] == "anomaly"]

    held_out = {
        session_id
        for sessions in HELD_OUT_NORMAL_SESSIONS.values()
        for session_id in sessions
    }
    train_pool = [row for row in normal if row["release_session_id"] not in held_out]
    train = sample_training_rows(train_pool, args.seed)
    normal_test_pool = [row for row in normal if row["release_session_id"] in held_out]
    normal_test = []
    for index, session_id in enumerate(sorted(held_out)):
        session_rows = [row for row in normal_test_pool if row["release_session_id"] == session_id]
        normal_test.extend(sample_timeline(session_rows, 100, args.seed + 50_000 + index))
    anomaly_test = []
    for index, fault in enumerate(TEST_FAULTS):
        fault_rows = [row for row in anomaly if row["fault_mode"] == fault]
        anomaly_test.extend(
            sample_balanced_sessions(fault_rows, 100, args.seed + 60_000 + index * 1_000)
        )
    test = normal_test + anomaly_test

    extra_fields = ["split_role", "sampling_seed"]
    output_fields = fields + extra_fields
    for row in train:
        row["split_role"] = "train_normal"
        row["sampling_seed"] = str(args.seed)
    for row in normal_test:
        row["split_role"] = "test_normal"
        row["sampling_seed"] = str(args.seed)
    for row in anomaly_test:
        row["split_role"] = "test_anomaly"
        row["sampling_seed"] = str(args.seed)

    train_ids = {row["clip_uid"] for row in train}
    test_ids = {row["clip_uid"] for row in test}
    missing_wavs = []
    for row in train + test:
        if not resolve_audio_path(args.release_root, row).is_file():
            missing_wavs.append(row["clip_audio_file"])

    train_counts = Counter(row["protocol_axis"] for row in train)
    if len(train) != 1000 or any(train_counts[c] != 250 for c in CONTEXTS):
        raise RuntimeError(f"invalid training composition: {dict(train_counts)}")
    if any(row["channel"] != "ch2" or row["benchmark_label"] != "normal" for row in train):
        raise RuntimeError("training manifest contains a non-front or non-normal row")
    if train_ids & test_ids:
        raise RuntimeError("train/test clip_uid leakage detected")
    if len(test) != 600 or len(test_ids) != 600:
        raise RuntimeError("test manifest must contain 600 unique clip_uid values")
    if Counter(row["benchmark_label"] for row in test) != {"normal": 300, "anomaly": 300}:
        raise RuntimeError("test manifest must contain 300 normal and 300 anomaly rows")
    if Counter(row["fault_mode"] for row in anomaly_test) != {fault: 100 for fault in TEST_FAULTS}:
        raise RuntimeError("test anomaly fault counts are not balanced")
    if missing_wavs:
        raise FileNotFoundError(f"missing {len(missing_wavs)} WAVs; first={missing_wavs[0]}")

    write_csv(args.output_dir / "run20_train.csv", output_fields, train)
    write_csv(args.output_dir / "run20_test.csv", output_fields, test)

    audit = {
        "status": "TRAIN_AND_TEST_SPLITS_VALIDATED",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "release_root": str(args.release_root.resolve()),
        "sampling_seed": args.seed,
        "channel": "ch2_front",
        "train_rows": len(train),
        "train_context_counts": dict(sorted(train_counts.items())),
        "train_session_counts": dict(sorted(Counter(row["release_session_id"] for row in train).items())),
        "held_out_normal_sessions": sorted(held_out),
        "train_only_normal_sessions": list(TRAIN_ONLY_NORMAL_SESSIONS),
        "test_rows": len(test),
        "test_label_counts": dict(sorted(Counter(row["benchmark_label"] for row in test).items())),
        "test_normal_session_counts": dict(sorted(Counter(row["release_session_id"] for row in normal_test).items())),
        "test_anomaly_fault_counts": dict(sorted(Counter(row["fault_mode"] for row in anomaly_test).items())),
        "test_anomaly_session_counts": dict(sorted(Counter(row["release_session_id"] for row in anomaly_test).items())),
        "train_test_clip_uid_overlap": len(train_ids & test_ids),
        "missing_wavs": len(missing_wavs),
        "test_manifest_locked": True,
    }
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
