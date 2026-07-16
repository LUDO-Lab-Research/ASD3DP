# Run: DCASE baseline autoencoder

This package contains three distinct layers:

- `source_snapshot/`: allow-listed, byte-identical files from the audited run;
- `public_reproduction/`: the same implementation made path-portable;
- `results/`: compact reported metrics and score evidence.

The complete local run directory was not copied because it contains generated
feature caches, checkpoints, staging symlinks, logs, and machine-local paths.
See the repository-level [`ERRATA_AND_PROVENANCE.md`](../ERRATA_AND_PROVENANCE.md)
for the evidence boundary.
