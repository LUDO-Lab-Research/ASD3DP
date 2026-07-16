# Run 20: ASD3DP DCASE AE Baseline

Date: 2026-07-12

## Summary

| Question | Answer |
| --- | --- |
| What does this run test? | Whether ASD3DP supports a complete normal-only ASD training and evaluation workflow |
| Training data | 1,000 domain-balanced normal clips |
| Test data | 300 held-out normal clips and 300 anomaly clips |
| Input | Front microphone (`ch2`), mono, 10 s WAV |
| Model | DCASE2023 Task 2-style autoencoder |
| Repetitions | Three model seeds: 23711, 23712, 23713 |
| Overall result | AUC 0.796 +/- 0.019; pAUC 0.722 +/- 0.009; F1 0.620 +/- 0.089 |
| Main observation | Toolhead collision is easy to rank; belt and extruder soft faults are harder |

This is a dataset functionality baseline. It is not a source/target experiment,
a domain-balancing ablation, or a state-of-the-art comparison.

## 1. Data Split

### Split overview

| Role | Normal | Anomaly | Channel | Separation rule |
| --- | ---: | ---: | --- | --- |
| Train | 1,000 | 0 | Front `ch2` | Normal-only; 16 sessions |
| Test | 300 | 300 | Front `ch2` | Held-out sessions; no train/test session or clip overlap |

### Normal operating contexts

| Normal context | Train | Test |
| --- | ---: | ---: |
| Slow loaded printing | 250 | 100 |
| Fast loaded printing | 250 | 100 |
| Slow no-filament motion | 250 | 0 |
| Fast no-filament motion | 250 | 100 |
| **Total** | **1,000** | **300** |

Training is domain-balanced by clip count: each of the four normal contexts
contributes exactly 250 clips. The test set has no slow-motion normal row
because no independent slow-motion session was available for holdout.

### Test anomaly composition

| Fault mode | Test anomaly clips |
| --- | ---: |
| Belt tension | 100 |
| Extruder failure | 100 |
| Toolhead collision | 100 |
| **Total** | **300** |

The split sampling seed is `13711`. Detailed session allocation and integrity
checks are in Appendix A.

## 2. Experiment Settings

| Setting | Value |
| --- | --- |
| Model | Fully connected DCASE2023 Task 2-style AE |
| Feature | 128-bin log-Mel; five frames per vector; 640-dimensional input |
| FFT / hop | 1,024 / 512 samples |
| Encoder | 128 -> 128 -> 128 -> 128 -> 8 |
| Decoder | 128 -> 128 -> 128 -> 128 -> 640 |
| Hidden layers | Linear -> BatchNorm -> ReLU |
| Training | Normal only; no validation |
| Batch / learning rate | 1,024 / 0.001 |
| Optimizer / epochs | Adam / 100 |
| Model seeds | 23711, 23712, 23713 |
| Checkpoints | Epochs 20, 40, 60, 80, 100 |
| Logging | W&B online |

Each 10 s file receives the mean reconstruction MSE over its feature vectors.
The reported metrics are file-level AUC, pAUC at maximum FPR 0.1, and F1 using
the seed-specific normal-score Gamma 90% threshold.

## 3. Results

### Overall

| Seed | AUC | pAUC | F1 | Gamma threshold |
| ---: | ---: | ---: | ---: | ---: |
| 23711 | 0.8167 | 0.7304 | 0.5172 | 14.1583 |
| 23712 | 0.7796 | 0.7125 | 0.6736 | 12.5356 |
| 23713 | 0.7906 | 0.7230 | 0.6689 | 12.5329 |
| **Mean +/- SD** | **0.7956 +/- 0.0191** | **0.7219 +/- 0.0090** | **0.6199 +/- 0.0890** | **13.0756 +/- 0.9376** |

### By fault mode

Each fault is evaluated against the same 300 normal test clips.

| Fault mode | Normal / anomaly | AUC | pAUC | F1 |
| --- | ---: | ---: | ---: | ---: |
| Belt tension | 300 / 100 | 0.723 +/- 0.024 | 0.592 +/- 0.016 | 0.295 +/- 0.189 |
| Extruder failure | 300 / 100 | 0.665 +/- 0.051 | 0.577 +/- 0.016 | 0.290 +/- 0.201 |
| Toolhead collision | 300 / 100 | 0.998 +/- 0.001 | 0.996 +/- 0.003 | 0.603 +/- 0.331 |

## 4. Interpretation

- The complete train-to-test pipeline runs successfully on the released data.
- Toolhead collision is almost perfectly ranked by the AE.
- Belt and extruder soft faults are substantially harder than collision.
- AUC and pAUC are more stable across seeds than Gamma-threshold F1.
- This run does not prove that domain-balanced sampling is better than simple
  random sampling because no random-sampling ablation was performed.

The reported F1 is not the highest F1 obtainable after inspecting test labels.
The overall test-label oracle is 0.731 +/- 0.021, compared with the reportable
Gamma F1 of 0.620 +/- 0.089. The oracle value is diagnostic only and must not
replace the label-free protocol result. Detailed threshold analysis is in
Appendix C.

## Appendix A. Detailed Split And Verification

### Training session allocation

| Normal context | Sessions | Allocation |
| --- | --- | --- |
| Slow loaded printing | `s0001`-`s0011` | 22-23 clips per session |
| Fast loaded printing | `s0013`, `s0014` | 125 clips per session |
| Slow no-filament motion | `s0016` | 250 clips from the only available training session |
| Fast no-filament motion | `s0017`, `s0018` | 125 clips per session |

Held-out normal sessions are `s0012` for slow printing, `s0015` for fast
printing, and `s0019` for fast motion.

| Verification | Result |
| --- | --- |
| Source manifest | ASD3DP v0.4.1 `event_clean_manifest.csv` |
| Manifest SHA-256 | `744537e34b2e51b993b23fbed6d029cb4d901562718e3518f0ff48354f08a692` |
| Train/test session overlap | 0 |
| Train/test clip-UID overlap | 0 |
| Missing staged WAVs | 0 |
| Cache/staging filename match | Complete |

Limitations of the split:

- Slow-motion training has only one session.
- The normal test set has no slow-motion holdout session.
- Microphone position is fixed to front `ch2`, not balanced across channels.

## Appendix B. Runtime And Checkpoints

| Item | Value |
| --- | --- |
| Cached training features | 934,000 x 640 float32 rows |
| Training storage | GPU-resident |
| Smoke epoch | Approximately 3.31 s after upload |
| Superseded memmap epoch | Approximately 156 s |
| Checkpoint state | Model, optimizer, epoch, seed, and RNG state |
| RNG state | Python, NumPy, Torch CPU, and all CUDA generators |

GPU-resident loading changes data delivery speed, not the cached feature
values, model, loss, optimizer, or batch size.

## Appendix C. F1 Threshold Diagnostic

| Scope | Gamma F1 | Test-label oracle F1 | Headroom |
| --- | ---: | ---: | ---: |
| Overall | 0.620 +/- 0.089 | 0.731 +/- 0.021 | +0.111 +/- 0.110 |
| Belt tension | 0.295 +/- 0.189 | 0.544 +/- 0.026 | +0.249 +/- 0.208 |
| Extruder failure | 0.290 +/- 0.201 | 0.471 +/- 0.049 | +0.181 +/- 0.249 |
| Toolhead collision | 0.603 +/- 0.331 | 0.992 +/- 0.003 | +0.388 +/- 0.332 |

Gamma is fitted to final-epoch train-normal batch reconstruction MSE values,
and its 90% quantile defines the protocol threshold. Oracle thresholds use test
labels and are therefore diagnostic upper bounds only.

## Appendix D. Evidence

Related documents:

- [Run 20 README](../README.md)
- [Locked experiment protocol](EXPERIMENT_PROTOCOL.md)

| Evidence | Path |
| --- | --- |
| Split audit | `../splits/split_audit.json` |
| Training manifest | `../splits/run20_train.csv` |
| Test manifest | `../splits/run20_test.csv` |
| Overall results | `../outputs/run20_seed_metrics.csv` |
| Fault results | `../outputs/run20_fault_summary_metrics.csv` |
| F1 diagnostic | `../outputs/run20_f1_threshold_diagnostic_summary.csv` |
| Per-file scores | `../outputs/seed_23711/file_scores.csv`, `../outputs/seed_23712/file_scores.csv`, `../outputs/seed_23713/file_scores.csv` |
| Runner | `../src/runner.py` |
