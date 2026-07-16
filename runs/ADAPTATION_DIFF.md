# Public reproduction adaptations

The source files under each `source_snapshot/` are byte-preserved. Changes
below apply only to `public_reproduction/`. Copied Python files in the public
reproduction trees also receive whitespace-only cleanup; executable behavior
is unchanged.

## DCASE baseline autoencoder

| File | Adaptation | Reason |
|---|---|---|
| `src/runner.py` | imports packaged `src/datasets.py` | remove the cross-run Run 03 import without changing loader behavior |
| `src/datasets.py` | byte-identical copy of the audited Run 03 dependency | make the run self-contained |
| `run.sh` | uses `python3`, requires `RELEASE_ROOT`, writes under `work/`, and uses offline W&B | remove machine-local defaults and account requirements |
| `scripts/stage_split.py` | refuses to delete an existing output unless `--replace-output` is passed | fail closed on destructive staging |
| result summarizers | read `../results` and the shared split directory | match the public layout |
| per-file score CSVs | removes the machine-local `path` column | publish score evidence without host paths |

## Pretrained encoder kNN

| File | Adaptation | Reason |
|---|---|---|
| `scripts/stage_asd3dp_pairs.py` | refuses to delete an existing output unless `--replace-output` is passed | fail closed on destructive staging |
| `src/datasets/prepare_dcase2026.py` | resolves the packaged ASD3DP data config by default | remove current-working-directory dependence |
| `config/data_config_asd3dp.yaml` | points at the package-local staged train/test directories | remove machine-local dataset paths |
| `config/run21_asd3dp.yaml` | uses package-local work/cache/output and external model directories | make paths portable |
| `scripts/download_sslam.py` | downloads the recorded upstream snapshot and verifies all consumed files | make the external model dependency explicit and fail on mismatch |
| `run.sh` | verifies SSLAM, stages the shared split, and runs the inspected config | provide one documented entrypoint |

No synchronization, annotation, split identity, score value, or reported
metric is changed by these adaptations.
