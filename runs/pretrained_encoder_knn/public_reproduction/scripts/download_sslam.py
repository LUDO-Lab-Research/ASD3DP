#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPOSITORY_ID = "ta012/SSLAM_pretrain"
EXPECTED_FILES = {
    "config.json": "2e7d67b4c77586cd495c9691152e49280c6be720a0e04da2c66822b1e94503ff",
    "configuration_eat.py": "761f19555f0d53aa7725ffb83a6e4eec652af97244134ea5147edca71be9b9e5",
    "eat_model.py": "9a836c65a7e02e7b3df80cca93324ff96d50b3c380a777c393803671e17faf37",
    "model_core.py": "f0157ab1d7f59e07db6da9d7845099dac7b818e1ea69322936da83364d8a2768",
    "model.safetensors": "8ec670adb241c710422ddd894ff1bade142ef0b25cf1ee68577aa45f89432298",
    "modeling_eat.py": "41eaf4b87923a56ef13899cbbe6d6ec9642d1f58db2e0b2661aae611eccfce9f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(destination: Path) -> None:
    errors = []
    for name, expected in EXPECTED_FILES.items():
        path = destination / name
        if not path.is_file():
            errors.append(f"missing {path}")
        elif sha256(path) != expected:
            errors.append(f"SHA-256 mismatch for {path}")
    if errors:
        raise RuntimeError("\n".join(errors))


def main() -> int:
    run_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=run_root / "external/sslam/SSLAM_pretrain",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=REPOSITORY_ID,
            local_dir=args.destination,
            allow_patterns=sorted(EXPECTED_FILES),
        )
    verify(args.destination)
    print(f"verified {REPOSITORY_ID} at {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
