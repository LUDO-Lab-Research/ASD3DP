# DCASE baseline autoencoder — public reproduction

This directory is the portable form of the code audited as the paper's DCASE
baseline AE. The byte-preserved inputs are under `../source_snapshot`; all
portability changes are listed in `../../ADAPTATION_DIFF.md`.

The authoritative split is shared by both published runs:

- split seed: `13711`;
- model seeds: `23711`, `23712`, and `23713`;
- train: 1,000 normal `ch2` clips from 16 sessions;
- test: 600 clips from 25 disjoint sessions.

## Run

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Point `RELEASE_ROOT` at the extracted ASD3DP release and run:

```bash
RELEASE_ROOT=/path/to/asd3dp_v0.4.1 bash ./run.sh
```

The launcher deliberately refuses to replace an existing `work/raw/3DPrinter`
staging directory. Move or remove that work directory before starting a clean
rerun. No W&B account is required; the reproduction launcher uses offline
logging.

## Verify

From the repository root:

```bash
python -m pytest -q runs/dcase_baseline_ae/public_reproduction/tests
```

The committed aggregate tables and path-sanitized per-file scores are under
`../results`. The score values, labels, basenames, and row counts are unchanged;
only the machine-local `path` column was removed.

Recompute the reported AUC, pAUC, and F1 for all three seeds:

```bash
python scripts/verify_reported_metrics.py
```
