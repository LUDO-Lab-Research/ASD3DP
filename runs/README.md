# ASD3DP paper runs

This directory publishes the exact paper split, compact result evidence, and
reviewed reproduction code for the two reported experiment families. Public
names are used instead of internal run numbers:

| Run | Inputs | Method | Status |
|---|---|---|---|
| [DCASE baseline AE](dcase_baseline_ae/) | front `ch2` | three-seed reconstruction AE | split and stored metrics validated |
| [Pretrained encoder k-NN](pretrained_encoder_knn/) | front `ch2`, rear `ch1` | frozen SSLAM L12; GenRep-style kNN and residual view | split reuse and stored metrics validated; historical source reconstruction provisional |

## Shared split

Both runs use the same ordered CSV manifests under
[`splits/dcase2026_seed13711/`](splits/dcase2026_seed13711/):

- split seed: `13711`;
- train: 1,000 normal clips, 16 sessions;
- test: 600 clips, 25 sessions;
- train/test clip overlap: `0`;
- train/test session overlap: `0`.

The AE model seeds are `23711`, `23712`, and `23713`. These are model seeds,
not the split seed.

## Reproduce and verify

Each run has a `source_snapshot/` and a separately reviewed
`public_reproduction/`. Start with the run-specific README. To verify the
repository artifact without starting training:

```bash
python -m pytest -q tests/test_runs_artifact.py
python -m pytest -q runs/dcase_baseline_ae/public_reproduction/tests
python -m pytest -q runs/pretrained_encoder_knn/public_reproduction/tests
```

`SOURCE_COPY_MANIFEST.csv` records the hashes of byte-preserved inputs.
`ADAPTATION_DIFF.md` records every portability change. Audio, feature caches,
model checkpoints, W&B data, and staging directories are intentionally not
stored in Git.

## Claim boundary

The split identities, session-disjointness, and metrics recomputed from stored
scores are validated. A byte-identical reconstruction of every historical
environment and every pretrained encoder k-NN execution-time source file is
not claimed. See
[`ERRATA_AND_PROVENANCE.md`](ERRATA_AND_PROVENANCE.md).
