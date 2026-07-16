# Pretrained encoder kNN — public reproduction

This is the portable reproduction of the paper's frozen SSLAM layer-12,
1-nearest-neighbor experiment. It exposes two linked views:

- **GenRep-style kNN:** front-only and rear-only frozen embeddings;
- **Residual view:** rear embedding minus `0.5 ×` front embedding.

Both views reuse the exact DCASE baseline AE split. They do not create a second
split and do not use G-code or telemetry as model inputs.

## Setup and run

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_sslam.py
RELEASE_ROOT=/path/to/asd3dp_v0.4.1 bash ./run.sh
```

The downloader uses the recorded Hugging Face repository and refuses to accept
the model unless all recorded SHA-256 values match. The staging command also
fails closed if `work/staging/ASD3DP` already exists.

## Verify

```bash
python -m pytest -q tests
```

The committed score evidence is under `../results`. The paper-facing table is
`run21_submitted_front_rear_table.csv`; right/left microphone diagnostics are
not presented as submitted paper systems in this package.

Recompute all three paper-facing metrics from the public score evidence:

```bash
python scripts/verify_reported_metrics.py
```

The complete three-system per-file table is explicitly labeled
`post_submission_verification`: it was regenerated from the preserved frozen
embedding cache because the historical compact per-file export omitted the
front-only rows.

This is a GenRep-style pipeline, not a claim that the upstream GenRepASD source
was executed byte-for-byte. External projects and the SSLAM model remain under
their own licenses and are linked in `../external_models.json`.
