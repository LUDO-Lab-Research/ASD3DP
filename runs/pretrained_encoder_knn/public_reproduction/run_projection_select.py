from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import hmean
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.prepare_dcase2026 import get_dcase2026  # noqa: E402
from src.residual_view.differet_view import collapse_paired_sequence  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class EncoderSpec:
    slug: str
    display: str
    model_name: str
    layer: int


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    label: str
    mode: str
    encoders: tuple[EncoderSpec, ...]


@dataclass(frozen=True)
class ProjectionConfig:
    config_path: Path
    train_split: str
    eval_split: str
    cache_root_template: str
    residual_alpha: float
    memmix_alpha: float
    n_mix_support: int
    prps_k: int
    projection_dim: int
    projection_seed: int
    decision_q: float
    score_dir: Path
    score_csv: Path | None
    write_submission: bool
    submission_dir: Path | None
    debug_prps: bool
    systems: tuple[SystemSpec, ...]


def log(message: str) -> None:
    print(message, flush=True)


def parse_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}.")


def require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a YAML mapping.")
    return value


def require_list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list.")
    return value


def load_yaml_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return require_mapping(config, str(path))


def load_encoder(item: object, index: int) -> EncoderSpec:
    encoder = require_mapping(item, f"encoders[{index}]")
    model_name = str(encoder["model_name"])
    layer = int(encoder["layer"])
    slug = str(encoder.get("slug", f"{model_name}_l{layer}"))
    return EncoderSpec(
        slug=slug,
        display=str(encoder.get("display", slug)),
        model_name=model_name,
        layer=layer,
    )


def load_systems(config: dict[str, object]) -> tuple[SystemSpec, ...]:
    systems = []
    for system_index, item in enumerate(require_list(config.get("systems"), "systems")):
        system = require_mapping(item, f"systems[{system_index}]")
        encoders = tuple(
            load_encoder(encoder, encoder_index)
            for encoder_index, encoder in enumerate(
                require_list(system.get("encoders"), f"systems[{system_index}].encoders")
            )
        )
        system_id = str(system["system_id"])
        systems.append(
            SystemSpec(
                system_id=system_id,
                label=str(system.get("label", system_id)),
                mode=str(system.get("mode", "prps")).lower(),
                encoders=encoders,
            )
        )
        if systems[-1].mode not in {"near", "far", "prps", "residual"}:
            raise ValueError(
                f"systems[{system_index}].mode must be 'near', 'far', 'prps', or 'residual'."
            )
    return tuple(systems)


def load_projection_config(config_path: Path) -> ProjectionConfig:
    config = load_yaml_config(config_path)
    score_csv = config.get("score_csv")
    submission_dir = config.get("submission_dir")
    return ProjectionConfig(
        config_path=config_path,
        train_split=str(config["train_split"]),
        eval_split=str(config["eval_split"]),
        cache_root_template=str(config["cache_root_template"]),
        residual_alpha=float(config.get("residual_alpha", 0.5)),
        memmix_alpha=float(config.get("memmix_alpha", config.get("alpha", 0.9))),
        n_mix_support=int(config.get("n_mix_support", 990)),
        prps_k=int(config.get("prps_k", 128)),
        projection_dim=int(config.get("projection_dim", 256)),
        projection_seed=int(config.get("projection_seed", 20260614)),
        decision_q=float(config.get("decision_q", 0.95)),
        score_dir=Path(config["score_dir"]),
        score_csv=Path(score_csv) if score_csv is not None else None,
        write_submission=parse_bool(
            config.get("write_submission", False),
            "write_submission",
        ),
        submission_dir=Path(submission_dir) if submission_dir is not None else None,
        debug_prps=parse_bool(config.get("debug_prps", False), "debug_prps"),
        systems=load_systems(config),
    )


def write_run_config(config: ProjectionConfig) -> None:
    config.score_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "train_split": config.train_split,
        "eval_split": config.eval_split,
        "cache_root_template": config.cache_root_template,
        "residual_alpha": config.residual_alpha,
        "memmix_alpha": config.memmix_alpha,
        "n_mix_support": config.n_mix_support,
        "prps_k": config.prps_k,
        "projection_dim": config.projection_dim,
        "projection_seed": config.projection_seed,
        "decision_q": config.decision_q,
        "write_submission": config.write_submission,
        "submission_dir": (
            str(config.submission_dir) if config.submission_dir is not None else None
        ),
        "debug_prps": config.debug_prps,
        "score_dir": str(config.score_dir),
        "score_csv": str(config.score_csv) if config.score_csv is not None else None,
        "systems": [
            {
                "system_id": system.system_id,
                "label": system.label,
                "mode": system.mode,
                "encoders": [asdict(encoder) for encoder in system.encoders],
            }
            for system in config.systems
        ],
    }
    with (config.score_dir / "resolved_config.yaml").open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    shutil.copyfile(config.config_path, config.score_dir / "input_config.yaml")


def hmean_or_nan(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return float("nan")
    if np.any(array < 0.0):
        raise ValueError(f"hmean received a negative value: {array}")
    return float(hmean(array))


def collapse_if_needed(values: np.ndarray, expected_len: int) -> np.ndarray:
    if len(values) == expected_len:
        return values
    if len(values) == expected_len * 2:
        return np.asarray(collapse_paired_sequence(values))
    raise ValueError(f"Cannot align sequence length {len(values)} to {expected_len}.")


def load_pair_features(
    spec: EncoderSpec,
    split: str,
    machine: str,
    cache_root_template: str,
) -> torch.Tensor:
    prefix = "train" if split in {"train", "dev_train", "eval_train"} else split
    try:
        cache_root = cache_root_template.format(model_name=spec.model_name)
    except KeyError as error:
        raise ValueError("cache_root_template may only use {model_name}.") from error
    path = (
        ROOT
        / cache_root
        / f"temp_dcase2026_{split}"
        / f"{prefix}_temporal_{machine}.pkl"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pair cache: {path}\n"
            "Generate it first with run_residual_view.py."
        )
    with path.open("rb") as handle:
        return pickle.load(handle).float()


def make_views(
    pair_features: torch.Tensor,
    layer: int,
    residual_alpha: float,
) -> dict[str, torch.Tensor]:
    layer_features = pair_features[layer - 1]
    near = layer_features[0::2].flatten(1).contiguous()
    far = layer_features[1::2].flatten(1).contiguous()
    residual = near - residual_alpha * far
    scale = (near * far).sum(dim=1, keepdim=True) / (
        (far * far).sum(dim=1, keepdim=True) + 1e-12
    )
    return {
        "near": near,
        "far": far,
        "residual": residual,
        "key": near - scale * far,
    }


def dist_matrix(x: torch.Tensor, y: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    x = x.to(DEVICE, non_blocking=True).float()
    y = y.to(DEVICE, non_blocking=True).float()
    y_t = y.T
    y_norm = (y**2).sum(dim=1, keepdim=True).T
    rows = []
    for start in range(0, x.shape[0], chunk):
        x_chunk = x[start:start + chunk]
        x_norm = (x_chunk**2).sum(dim=1, keepdim=True)
        dist_sq = x_norm + y_norm - 2 * x_chunk @ y_t
        rows.append(torch.sqrt(torch.clamp(dist_sq, min=0.0)).cpu())
    return torch.cat(rows, dim=0)


def nearest_scores(test_features: torch.Tensor, bank: torch.Tensor) -> np.ndarray:
    dist = dist_matrix(test_features, bank)
    return torch.topk(dist, k=1, dim=1, largest=False).values.numpy().reshape(-1)


def project(
    features: torch.Tensor,
    output_dim: int,
    projection_seed: int,
) -> torch.Tensor:
    features = features.to(DEVICE, non_blocking=True).float()
    generator = torch.Generator(device=DEVICE).manual_seed(projection_seed)
    projection = torch.randn(
        features.shape[1],
        output_dim,
        generator=generator,
        device=DEVICE,
        dtype=torch.float32,
    )
    projection = projection / (features.shape[1] ** 0.5)
    return (features @ projection).cpu()


def kcenter_order(features: torch.Tensor, k: int) -> torch.Tensor:
    features = features.to(DEVICE, non_blocking=True).float()
    k = min(k, features.shape[0])
    if k <= 0:
        raise ValueError(f"prps_k must be positive, got {k}.")

    first = torch.argmin(((features - features.mean(dim=0, keepdim=True)) ** 2).sum(dim=1)).item()
    selected = [first]
    min_dist = ((features - features[first]) ** 2).sum(dim=1)
    for _ in range(1, k):
        idx = torch.argmax(min_dist).item()
        selected.append(idx)
        min_dist = torch.minimum(min_dist, ((features - features[idx]) ** 2).sum(dim=1))
    return torch.tensor(selected, dtype=torch.long)


def make_memmix(
    train_features: torch.Tensor,
    train_domains: np.ndarray,
    memmix_alpha: float,
    n_mix_support: int,
) -> torch.Tensor:
    if n_mix_support <= 0:
        return train_features.new_empty((0, train_features.shape[1]))

    source_idx = torch.from_numpy(np.flatnonzero(train_domains == "source")).long()
    target_idx = torch.from_numpy(np.flatnonzero(train_domains == "target")).long()
    if len(source_idx) == 0 or len(target_idx) == 0:
        raise ValueError("MemMix requires both source and target samples.")

    source = train_features[source_idx]
    target = train_features[target_idx]
    _, topk = torch.topk(
        dist_matrix(target, source),
        k=min(n_mix_support, source.shape[0]),
        dim=1,
        largest=False,
    )
    mixed = [
        memmix_alpha * target[target_i] + (1.0 - memmix_alpha) * source[source_indices]
        for target_i, source_indices in enumerate(topk)
    ]
    return torch.cat(mixed, dim=0)


def loo_scores(
    train_features: torch.Tensor,
    bank_features: torch.Tensor,
    bank_indices: torch.Tensor | None = None,
) -> np.ndarray:
    dist = dist_matrix(train_features, bank_features)
    if bank_indices is None:
        dist[:, : train_features.shape[0]].fill_diagonal_(float("inf"))
    else:
        index_to_col = {int(idx): col for col, idx in enumerate(bank_indices.tolist())}
        rows = []
        cols = []
        for row_idx in range(train_features.shape[0]):
            col = index_to_col.get(row_idx)
            if col is not None:
                rows.append(row_idx)
                cols.append(col)
        if rows:
            dist[
                torch.tensor(rows, dtype=torch.long),
                torch.tensor(cols, dtype=torch.long),
            ] = float("inf")
    return torch.topk(dist, k=1, dim=1, largest=False).values.numpy().reshape(-1)


def machine_meta(
    train_split: str,
    eval_split: str,
    machine: str,
) -> dict[str, np.ndarray]:
    datasets = get_dcase2026(train_split)
    if eval_split not in datasets:
        raise ValueError(f"Missing eval_split={eval_split} for train_split={train_split}.")

    train_data = datasets["train"]
    test_data = datasets[eval_split]
    train_mask = np.asarray(train_data["machine_names"]) == machine
    test_mask = np.asarray(test_data["machine_names"]) == machine
    return {
        "train_domains": np.asarray(train_data["source_list"])[train_mask],
        "test_files": np.asarray(test_data["file_list"])[test_mask],
        "test_labels": np.asarray(test_data["label_list"])[test_mask],
        "test_domains": np.asarray(test_data["source_list"])[test_mask],
    }


def scores_for_encoder(
    spec: EncoderSpec,
    config: ProjectionConfig,
    mode: str,
    machine: str,
) -> tuple[np.ndarray, np.ndarray]:
    meta = machine_meta(config.train_split, config.eval_split, machine)
    train_views = make_views(
        load_pair_features(spec, config.train_split, machine, config.cache_root_template),
        spec.layer,
        config.residual_alpha,
    )
    test_views = make_views(
        load_pair_features(spec, config.eval_split, machine, config.cache_root_template),
        spec.layer,
        config.residual_alpha,
    )

    view_name = mode if mode in {"near", "far"} else "residual"
    train_residual = train_views[view_name]
    test_residual = test_views[view_name]
    train_domains = collapse_if_needed(
        np.asarray(meta["train_domains"]),
        train_residual.shape[0],
    )

    memmix_bank = (
        make_memmix(
            train_residual,
            train_domains,
            config.memmix_alpha,
            config.n_mix_support,
        )
        if config.n_mix_support > 0
        else train_residual.new_empty((0, train_residual.shape[1]))
    )
    full_train_bank = torch.cat(
        [
            train_residual,
            memmix_bank,
        ],
        dim=0,
    )

    # No test-time calibration: test clips are only queried for clip-level
    # anomaly scores against banks built from train data.
    residual_test_scores = nearest_scores(test_residual, full_train_bank)
    residual_train_scores = loo_scores(train_residual, full_train_bank)

    if mode in {"near", "far", "residual"}:
        return residual_test_scores, residual_train_scores

    # PRPS prototype selection is train-only.
    prototype_indices = kcenter_order(
        project(train_views["key"], config.projection_dim, config.projection_seed),
        config.prps_k,
    )
    if config.debug_prps:
        debug_dir = config.score_dir / "debug_prps"
        debug_dir.mkdir(parents=True, exist_ok=True)
        selected_domains = np.asarray(train_domains)[prototype_indices.numpy()]
        selected_source = int(np.sum(selected_domains == "source"))
        selected_target = int(np.sum(selected_domains == "target"))
        pd.DataFrame(
            {
                "rank": np.arange(len(prototype_indices)),
                "train_index": prototype_indices.numpy(),
                "domain": selected_domains,
            }
        ).to_csv(
            debug_dir / f"{machine}_{spec.slug}_seed{config.projection_seed}_indices.csv",
            index=False,
        )
        log(
            f"[prps-debug] {machine} {spec.slug} seed={config.projection_seed}: "
            f"source={selected_source} target={selected_target} "
            f"k={len(prototype_indices)}"
        )
    prototype_bank = train_residual[prototype_indices]
    prototype_test_scores = nearest_scores(test_residual, prototype_bank)
    prototype_train_scores = loo_scores(
        train_residual,
        prototype_bank,
        prototype_indices,
    )

    return (
        0.5 * residual_test_scores + 0.5 * prototype_test_scores,
        0.5 * residual_train_scores + 0.5 * prototype_train_scores,
    )


def system_scores(
    system: SystemSpec,
    config: ProjectionConfig,
    machine: str,
) -> tuple[np.ndarray, np.ndarray]:
    test_scores = []
    train_scores = []
    for spec in system.encoders:
        log(
            f"[score] system={system.label} encoder={spec.display} "
            f"mode={system.mode.upper()} machine={machine}"
        )
        test, train = scores_for_encoder(spec, config, system.mode, machine)
        test_scores.append(test)
        train_scores.append(train)
    return np.mean(np.stack(test_scores), axis=0), np.mean(np.stack(train_scores), axis=0)


def eval_score(gt_list: np.ndarray, scores: np.ndarray) -> tuple[float, ...]:
    gt_list = np.asarray(gt_list)
    roc_curve(gt_list, scores)
    auc = roc_auc_score(gt_list, scores)
    pauc = roc_auc_score(gt_list, scores, max_fpr=0.1)
    precision, recall, _ = precision_recall_curve(gt_list, scores)
    f1_scores = 2 * precision * recall / (
        precision + recall + np.finfo(float).eps
    )
    return float(auc), float(pauc), float(f1_scores[np.argmax(f1_scores)])


def metric_row(
    system: SystemSpec,
    machine: str,
    meta: dict[str, np.ndarray],
    scores: np.ndarray,
    config: ProjectionConfig,
) -> dict[str, object]:
    test_labels = collapse_if_needed(np.asarray(meta["test_labels"]), len(scores))
    test_domains = collapse_if_needed(np.asarray(meta["test_domains"]), len(scores))
    has_eval_labels = set(np.unique(test_labels).tolist()).issubset({0, 1}) and (
        len(np.unique(test_labels)) == 2
    )

    row = {
        "layer": system.system_id,
        "system": system.label,
        "machine": machine,
        "train_split": config.train_split,
        "eval_split": config.eval_split,
        "auc_source": np.nan,
        "auc_target": np.nan,
        "pauc": np.nan,
        "official_score": np.nan,
        "auc": np.nan,
        "pauc_source": np.nan,
        "pauc_target": np.nan,
        "f1": np.nan,
        "f1_source": np.nan,
        "f1_target": np.nan,
        "n_scores": len(scores),
        "prps_k": config.prps_k,
        "projection_dim": config.projection_dim,
        "projection_seed": config.projection_seed,
    }
    if not has_eval_labels:
        return row

    source_mask = np.asarray(test_domains) == "source"
    known_domain_mask = np.isin(test_domains, ["source", "target"])
    anomaly_mask = np.asarray(test_labels) != 0
    source_eval_mask = source_mask | anomaly_mask
    target_eval_mask = (~source_mask) | anomaly_mask

    auc_all, pauc_all, f1_all = eval_score(test_labels, scores)
    if not np.any(known_domain_mask):
        row.update(
            {
                "pauc": pauc_all,
                "auc": auc_all,
                "f1": f1_all,
            }
        )
        return row
    auc_source, pauc_source, f1_source = eval_score(
        test_labels[source_eval_mask],
        scores[source_eval_mask],
    )
    auc_target, pauc_target, f1_target = eval_score(
        test_labels[target_eval_mask],
        scores[target_eval_mask],
    )
    row.update(
        {
            "auc_source": auc_source,
            "auc_target": auc_target,
            "pauc": pauc_all,
            "official_score": float(hmean([auc_source, auc_target, pauc_all])),
            "auc": auc_all,
            "pauc_source": pauc_source,
            "pauc_target": pauc_target,
            "f1": f1_all,
            "f1_source": f1_source,
            "f1_target": f1_target,
        }
    )
    return row


def official_score_from_rows(df_log: pd.DataFrame, layer: object, system: str) -> float:
    mask = (df_log["layer"] == layer) & (df_log["system"] == system)
    values = df_log.loc[mask, ["auc_source", "auc_target", "pauc"]].to_numpy(dtype=float).reshape(-1)
    values = values[~np.isnan(values)]
    return float(hmean(values)) if values.size > 0 else float("nan")


def write_summary_files(
    rows: list[dict[str, object]],
    config: ProjectionConfig,
) -> None:
    config.score_dir.mkdir(parents=True, exist_ok=True)
    df_log = pd.DataFrame(rows)
    df_log.to_csv(config.score_dir / "result.csv", index=False)

    metric_cols = [
        "auc_source",
        "auc_target",
        "pauc",
        "official_score",
        "auc",
        "pauc_source",
        "pauc_target",
        "f1",
        "f1_source",
        "f1_target",
    ]
    df_avg = (
        df_log.groupby(["layer", "system"])[metric_cols]
        .agg(hmean_or_nan)
        .reset_index()
    )
    df_avg["official_score"] = [
        official_score_from_rows(df_log, row["layer"], str(row["system"]))
        for _, row in df_avg.iterrows()
    ]
    df_avg["oc"] = df_avg["official_score"]
    df_avg.to_csv(config.score_dir / "df_avg.csv", index=False)

    n_machines = df_log["machine"].nunique()
    mode_by_system = {system.label: system.mode for system in config.systems}
    for _, row in df_avg.iterrows():
        log(
            f"[projection-summary] {row['system']} "
            f"mode={mode_by_system[str(row['system'])].upper()} "
            f"machines={n_machines}: "
            f"sAUC={row['auc_source']:.4f} "
            f"tAUC={row['auc_target']:.4f} "
            f"pAUC={row['pauc']:.4f} "
            f"DCASE-official={row['oc']:.4f}"
        )

    selection_metric = (
        "official_score"
        if df_log["official_score"].notna().any()
        else "auc"
    )
    if df_avg[selection_metric].notna().any():
        best_row = df_avg.loc[[df_avg[selection_metric].idxmax()]]
        best_layer = best_row.iloc[0]["layer"]
        best_system = str(best_row.iloc[0]["system"])
        machine_rows = df_log[
            (df_log["layer"] == best_layer)
            & (df_log["system"] == best_system)
        ]
        machine_rows[
            [
                "machine",
                "auc",
                "pauc",
                "auc_source",
                "auc_target",
                "official_score",
                "system",
            ]
        ].to_csv(config.score_dir / "machine_layer_wise.csv", index=False)
        best_row.to_csv(config.score_dir / "final_best_layer_wise.csv", index=False)
        df_log.loc[df_log.groupby("machine")[selection_metric].idxmax()].to_csv(
            config.score_dir / "best_each_machine.csv",
            index=False,
        )
    else:
        for filename in [
            "machine_layer_wise.csv",
            "final_best_layer_wise.csv",
            "best_each_machine.csv",
        ]:
            pd.DataFrame().to_csv(config.score_dir / filename, index=False)

    if config.score_csv is not None:
        config.score_csv.parent.mkdir(parents=True, exist_ok=True)
        df_avg.to_csv(config.score_csv, index=False)


def write_submission_files(
    system: SystemSpec,
    machine: str,
    meta: dict[str, np.ndarray],
    scores: np.ndarray,
    train_scores: np.ndarray,
    config: ProjectionConfig,
) -> None:
    if config.submission_dir is None:
        raise ValueError("write_submission=True requires submission_dir.")

    system_dir = config.submission_dir / system.label
    system_dir.mkdir(parents=True, exist_ok=True)
    files = [
        Path(path).name
        for path in collapse_if_needed(np.asarray(meta["test_files"]), len(scores))
    ]
    threshold = float(np.quantile(train_scores, config.decision_q))
    decisions = (scores > threshold).astype(int)

    score_path = system_dir / f"anomaly_score_{machine}_section_00_test.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for filename, value in zip(files, scores):
            writer.writerow([filename, f"{float(value):.8f}"])

    decision_path = system_dir / f"decision_result_{machine}_section_00_test.csv"
    with decision_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for filename, value in zip(files, decisions):
            writer.writerow([filename, str(int(value))])

    meta_source = (
        ROOT / "submissions" / "task2" / system.label / f"{system.label}.meta.yaml"
    )
    if meta_source.exists():
        shutil.copy2(meta_source, system_dir / f"{system.label}.meta.yaml")

    log(
        f"[submission] {system.label} {machine}: "
        f"q={config.decision_q} threshold={threshold:.6f} "
        f"ones={int(decisions.sum())} dir={system_dir}"
    )


def run(config: ProjectionConfig) -> None:
    train_data = get_dcase2026(config.train_split)["train"]
    machine_names = sorted(np.unique(train_data["machine_names"]).tolist())
    rows = []

    log(f"[device] {DEVICE}")
    log(f"[split] train_split={config.train_split} eval_split={config.eval_split}")
    log(f"[machines] {machine_names}")
    write_run_config(config)

    for system in config.systems:
        for machine in machine_names:
            meta = machine_meta(config.train_split, config.eval_split, machine)
            scores, train_scores = system_scores(system, config, machine)
            row = metric_row(system, machine, meta, scores, config)
            rows.append(row)
            if config.write_submission:
                write_submission_files(system, machine, meta, scores, train_scores, config)
            log(
                f"[result] {system.label} {machine} mode={system.mode.upper()}: "
                f"sAUC={row['auc_source']:.4f} "
                f"tAUC={row['auc_target']:.4f} "
                f"pAUC={row['pauc']:.4f} "
                f"official={row['official_score']:.4f}"
            )

    write_summary_files(rows, config)


def parse_args() -> ProjectionConfig:
    parser = argparse.ArgumentParser(
        description="Score cached residual-view pair embeddings from one system YAML."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    return load_projection_config(args.config)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
