#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RUN_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))
from datasets import ChunkedFeatureDataset, FeatureConfig, build_feature_store  # noqa: E402


CHECKPOINT_EPOCHS = (20, 40, 60, 80, 100)


class AutoEncoder(nn.Module):
    """Exact DCASE2023T2-AE topology and BatchNorm settings."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(128, 128), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(128, 128), requires_grad=False)
        encoder = []
        previous = input_dim
        for width in (128, 128, 128, 128, 8):
            encoder.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width, momentum=0.01, eps=1e-03),
                    nn.ReLU(),
                ]
            )
            previous = width
        decoder = []
        previous = 8
        for width in (128, 128, 128, 128):
            decoder.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width, momentum=0.01, eps=1e-03),
                    nn.ReLU(),
                ]
            )
            previous = width
        decoder.append(nn.Linear(previous, input_dim))
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x.view(-1, self.input_dim))
        return self.decoder(encoded)


def reduce_file_score(frame_vector_mse: np.ndarray) -> float:
    return float(np.mean(np.asarray(frame_vector_mse, dtype=np.float64)))


def fit_gamma_threshold(scores: np.ndarray, quantile: float = 0.9) -> tuple[list[float], float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Gamma fitting requires at least two finite normal scores")
    params = [float(value) for value in scipy.stats.gamma.fit(values)]
    return params, gamma_quantile(params, quantile)


def gamma_quantile(params: list[float], quantile: float) -> float:
    shape, location, scale = params
    return float(scipy.stats.gamma.ppf(quantile, a=shape, loc=location, scale=scale))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100, choices=[100])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--chunk-files", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--train-storage", default="gpu", choices=["gpu", "memory", "memmap"])
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline"])
    parser.add_argument("--smoke-one-epoch", action="store_true")
    return parser


def effective_epochs(args: argparse.Namespace) -> int:
    return 1 if args.smoke_one_epoch else args.epochs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def checkpoint_payload(model, optimizer, epoch, input_dim, seed):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "input_dim": input_dim,
        "seed": seed,
        "rng_state": capture_rng_state(),
    }


def load_contiguous_features(store_dir: Path) -> np.ndarray:
    store_dir = Path(store_dir)
    done = json.loads((store_dir / "DONE.json").read_text())
    expected_rows = int(done["rows"])
    expected_dims = int(done["config"]["n_mels"]) * int(done["config"]["frames"])
    chunks = sorted(store_dir.glob("chunk_*.npy"))
    if len(chunks) != int(done["chunks"]):
        raise ValueError(f"chunk count mismatch: expected {done['chunks']}, found {len(chunks)}")
    features = np.empty((expected_rows, expected_dims), dtype=np.float32)
    offset = 0
    for path in chunks:
        chunk = np.load(path, mmap_mode="r")
        if chunk.ndim != 2 or chunk.shape[1] != expected_dims:
            raise ValueError(
                f"feature dimension mismatch in {path.name}: expected {expected_dims}, found {chunk.shape}"
            )
        end = offset + len(chunk)
        if end > expected_rows:
            raise ValueError(f"row count mismatch: cache exceeds declared {expected_rows} rows")
        features[offset:end] = chunk
        offset = end
    if offset != expected_rows:
        raise ValueError(f"row count mismatch: expected {expected_rows}, loaded {offset}")
    return features


def iter_resident_batches(features: torch.Tensor, batch_size: int, device: torch.device):
    order = torch.randperm(len(features), device="cpu")
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size].to(device, non_blocking=True)
        yield features.index_select(0, indices)


@torch.no_grad()
def score_feature_rows(model, features: np.ndarray, batch_size: int, device) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(features), batch_size):
        batch = np.array(features[start : start + batch_size], dtype=np.float32, copy=True)
        x = torch.from_numpy(batch).to(device)
        reconstruction = model(x)
        parts.append(torch.mean((reconstruction - x) ** 2, dim=1).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def score_files(model, store_dir: Path, batch_size: int, device) -> list[dict[str, object]]:
    with (store_dir / "manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    by_chunk: dict[str, list[dict[str, str]]] = {}
    for row in manifest:
        by_chunk.setdefault(row["chunk"], []).append(row)
    output = []
    for chunk_name in sorted(by_chunk):
        features = np.load(store_dir / chunk_name, mmap_mode="r")
        row_scores = score_feature_rows(model, features, batch_size, device)
        for row in by_chunk[chunk_name]:
            start = int(row["chunk_row_start"])
            count = int(row["row_count"])
            output.append(
                {
                    "basename": row["basename"],
                    "path": row["path"],
                    "label": int(row["label"]),
                    "score": reduce_file_score(row_scores[start : start + count]),
                    "row_count": count,
                }
            )
    return output


def main() -> int:
    args = build_parser().parse_args()
    epochs = effective_epochs(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    config = FeatureConfig()
    train_store = build_feature_store(
        args.raw_root, args.cache_root, "train", config,
        chunk_files=args.chunk_files, force=args.force_cache,
    )
    test_store = build_feature_store(
        args.raw_root, args.cache_root, "test", config,
        chunk_files=args.chunk_files, force=args.force_cache,
    )
    resident_train = None
    if args.train_storage in {"gpu", "memory"}:
        train_features = load_contiguous_features(train_store)
        input_dim = int(train_features.shape[1])
        if args.train_storage == "gpu":
            resident_train = torch.from_numpy(train_features).to(device)
            train_dataset = None
            train_loader = None
        else:
            train_dataset = TensorDataset(torch.from_numpy(train_features))
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
            )
    else:
        train_dataset = ChunkedFeatureDataset(train_store)
        input_dim = train_dataset.dims
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
    model = AutoEncoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if int(payload["seed"]) != args.seed:
            raise ValueError(f"checkpoint seed {payload['seed']} does not match requested seed {args.seed}")
        if int(payload["input_dim"]) != input_dim:
            raise ValueError("checkpoint input dimension does not match the active feature store")
        if "rng_state" not in payload:
            raise ValueError("checkpoint has no RNG state and cannot be resumed exactly")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch"]) + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        group="asd3dp_run20_dcase_ae",
        name=f"run20_seed_{args.seed}",
        config={**vars(args), "validation": "none", "checkpoint_epochs": CHECKPOINT_EPOCHS},
    )
    history_path = args.output_dir / "history.csv"
    if start_epoch == 1:
        with history_path.open("w") as handle:
            handle.write("epoch,train_loss,epoch_seconds\n")
    elif not history_path.is_file():
        with history_path.open("w") as handle:
            handle.write("epoch,train_loss,epoch_seconds\n")

    final_epoch_batch_scores = []
    for epoch in range(start_epoch, epochs + 1):
        started = time.monotonic()
        model.train()
        batch_scores = []
        epoch_batches = (
            iter_resident_batches(resident_train, args.batch_size, device)
            if resident_train is not None
            else train_loader
        )
        for x in epoch_batches:
            if resident_train is None:
                if isinstance(x, (tuple, list)):
                    x = x[0]
                x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(x)
            loss = torch.mean((reconstruction - x) ** 2)
            loss.backward()
            optimizer.step()
            batch_scores.append(float(loss.detach()))
        train_loss = float(np.mean(batch_scores))
        elapsed = time.monotonic() - started
        with history_path.open("a") as handle:
            handle.write(f"{epoch},{train_loss:.10f},{elapsed:.6f}\n")
        run.log({"epoch": epoch, "train/loss": train_loss, "time/epoch_seconds": elapsed}, step=epoch)
        if epoch in CHECKPOINT_EPOCHS:
            path = args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(checkpoint_payload(model, optimizer, epoch, input_dim, args.seed), path)
            run.log({"epoch": epoch, "checkpoint/saved": 1}, step=epoch)
        if epoch == epochs:
            final_epoch_batch_scores = batch_scores

    final_path = args.output_dir / f"model_epoch_{epochs:03d}.pt"
    torch.save(checkpoint_payload(model, optimizer, epochs, input_dim, args.seed), final_path)
    gamma_params, threshold = fit_gamma_threshold(np.asarray(final_epoch_batch_scores), 0.9)
    gamma_record = {
        "fit_scores": "final_epoch_train_normal_batch_reconstruction_mse",
        "validation_scores_included": False,
        "quantile": 0.9,
        "shape": gamma_params[0],
        "location": gamma_params[1],
        "scale": gamma_params[2],
        "threshold": threshold,
        "score_count": len(final_epoch_batch_scores),
        "smoke_only": args.smoke_one_epoch,
    }
    (args.output_dir / "gamma_threshold.json").write_text(json.dumps(gamma_record, indent=2) + "\n")

    test_rows = score_files(model, test_store, args.batch_size, device)
    labels = np.asarray([row["label"] for row in test_rows], dtype=np.int64)
    scores = np.asarray([row["score"] for row in test_rows], dtype=np.float64)
    decisions = (scores > threshold).astype(np.int64)
    metrics = {
        "auc": float(roc_auc_score(labels, scores)),
        "pauc": float(roc_auc_score(labels, scores, max_fpr=0.1)),
        "f1": float(f1_score(labels, decisions)),
        "threshold": threshold,
        "n_files": len(test_rows),
        "n_normal": int(np.sum(labels == 0)),
        "n_anomaly": int(np.sum(labels == 1)),
    }
    with (args.output_dir / "file_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["basename", "path", "label", "score", "row_count"])
        writer.writeheader()
        writer.writerows(test_rows)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    run.log({f"eval/{key}": value for key, value in metrics.items()})
    run.summary.update(metrics)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
