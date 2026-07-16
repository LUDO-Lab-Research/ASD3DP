# Run 20: Single DCASE AE Baseline

- [Complete split, training, result, and F1 report](docs/RUN20_REPORT.md)

## Completed result

The three-seed Run 20 evaluation is complete. Seed-level metrics are in
[`outputs/run20_seed_metrics.csv`](outputs/run20_seed_metrics.csv), and the
mean and sample standard deviation are in
[`outputs/run20_summary_metrics.csv`](outputs/run20_summary_metrics.csv).
These results apply only to the locked front-channel Run 20 split
(`splits/run20_train.csv`, `splits/run20_test.csv`).

Fault-wise comparisons use the same 300 test-normal clips against 100 anomaly
clips from one fault mode at a time. Seed-level values are in
[`outputs/run20_fault_seed_metrics.csv`](outputs/run20_fault_seed_metrics.csv),
and their mean and sample standard deviation are in
[`outputs/run20_fault_summary_metrics.csv`](outputs/run20_fault_summary_metrics.csv).

- [Locked experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Validated training split](splits/run20_train.csv)
- [Validated test split](splits/run20_test.csv)
- [Split audit](splits/split_audit.json)
- [Run 20 runner](src/runner.py)
- [Three-seed launcher](run.sh)

Safety status: split audit and 1-epoch end-to-end smoke validated; all three
100-epoch runs completed.

Smoke artifacts: [seed 23711](outputs/smoke_seed_23711/). Smoke metrics are
diagnostic only and must not be reported as experiment results.

Superseded design:

- [Runs 10-12 source/target protocol](../10_12_asd3dp_baselines/docs/EXPERIMENT_PROTOCOL.md)
- [Composite split v2](../10_12_asd3dp_baselines/splits_v2/SPLIT_LOCK.md)
