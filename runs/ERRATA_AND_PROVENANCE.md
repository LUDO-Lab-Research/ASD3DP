# Errata and provenance

## Validated evidence

- The exact ordered train and test CSVs reproduce from the v0.4.1 locked
  event-clean manifest with split seed `13711`.
- The 1,000 training clips and 600 test clips have zero clip and session
  overlap.
- The DCASE baseline AE stored AUC, pAUC, and F1 values recompute from its stored per-file
  scores for model seeds `23711`, `23712`, and `23713`.
- The pretrained encoder k-NN reuses the same clip identities and row order. Its paper-facing
  front-only, rear-only, and rear-minus-front metrics recompute exactly from a
  post-submission projection of the preserved frozen embedding cache.
- The public reproduction tests are copied from the audited runs and augmented
  only with portability and fail-closed staging checks.

## Historical boundaries

- The experiment workspace was not itself a Git repository at execution time.
  `SOURCE_COPY_MANIFEST.csv` therefore locks file hashes rather than inventing
  a historical execution commit.
- The DCASE baseline AE runner imported `datasets.py` from an earlier internal
  G-code-aligned experiment.
  The exact dependency is preserved under `source_snapshot/shared/` and copied
  beside the public runner for portable imports.
- The original DCASE baseline AE shell launcher is not published byte-for-byte because it
  contains machine-local defaults and a W&B entity. Its training arguments are
  preserved in the run report and the portable launcher.
- The original DCASE baseline AE per-file scores contained a machine-local staging path.
  Public score files remove only that path column; basename, label, score, and
  row count are unchanged.
- The pretrained encoder k-NN rear staging predates the generalized channel staging source. The
  audited clip identities and staged audio properties are validated, but the
  complete execution-time rear source is not available as a byte-identical
  file snapshot.
- The current pretrained encoder k-NN source includes later portability and front-only support.
  It reproduces the audited score calculations but is not represented as an
  untouched historical execution tree.
- The original compact pretrained encoder k-NN per-file table contained rear-only and residual
  scores but not front-only scores. `results/post_submission_verification/`
  was regenerated from the preserved frozen embedding cache and current
  audited projection code; it is verification evidence, not an execution-time
  output snapshot.
- The pretrained encoder k-NN environment specification is partially pinned. The public
  reproduction keeps the inspected requirement ranges and does not fabricate
  a historical lock file.

## Excluded generated material

Audio staging, mel and embedding caches, checkpoints, pretrained weights,
Python caches, PID files, W&B state, failed setup outputs, and superseded
diagnostics remain outside the Git repository. SSLAM is downloaded from its
upstream model repository and accepted only after checksum verification.
