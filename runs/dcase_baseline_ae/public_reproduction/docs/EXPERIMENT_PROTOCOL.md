# ASD3DP Run 20 Experiment Protocol

Status: 1-EPOCH END-TO-END SMOKE VALIDATED; 100-EPOCH RUNS NOT STARTED

## Locked Protocol

| Item | Locked value |
|---|---|
| Dataset table | Domain/condition table retained as dataset composition and analysis axes |
| Model | Exact repository `DCASE2023T2-AE` architecture |
| Features and scoring | Exact `DCASE2023T2-AE` feature extraction and anomaly scoring |
| Input | Mono WAV |
| Training data | Exactly 1,000 normal WAV files |
| Training sampling | Domain-balanced random sampling |
| Domain attributes | motion, speed, microphone position |
| Insufficient normal pool | Expand to the nearest available normal context |
| Validation | None |
| Anomaly use | Test only |
| Epochs | 100 |
| Batch size | 1,024 |
| Learning rate | 0.001 |
| Checkpoints | Epochs 20, 40, 60, 80, and 100 |
| Logging | W&B mandatory |
| Model seeds | `23711`, `23712`, `23713` |
| Training jobs | 3 |
| Overall metrics | AUC, pAUC, F1 |
| F1 threshold | 90% quantile of a Gamma distribution fitted to normal scores, following the DCASE method |
| Post-hoc analysis | fault mode, microphone, motion, speed |

Domain balancing is required because an imbalanced normal pool can let the
model learn operating-domain differences, such as slow versus fast operation,
instead of anomaly evidence. Sampling strata and realized counts must be
written into the new manifest audit before training. If a stratum does not
contain enough normal clips, sampling expands to the nearest normal context;
the expansion order and realized composition must be recorded.

| Training context | WAVs | Sessions |
|---|---:|---|
| Slow loaded printing | 250 | `s0001`-`s0011` |
| Fast loaded printing | 250 | `s0013`, `s0014` |
| Slow no-filament motion | 250 | `s0016` (train-only) |
| Fast no-filament motion | 250 | `s0017`, `s0018` |

All training WAVs use front `ch2` and are fully contained in
`printing_motion`. Within each context, quotas are equal across sessions and
samples cover each session timeline. Normal sessions `s0012`, `s0015`, and
`s0019` are held out as test candidates.

| Test group | WAVs |
|---|---:|
| Normal: `s0012` | 100 |
| Normal: `s0015` | 100 |
| Normal: `s0019` | 100 |
| Belt-tension anomaly | 100 |
| Extruder anomaly | 100 |
| Toolhead-collision anomaly | 100 |

The 600 test WAVs all use front `ch2`. Anomaly rows are balanced across fault
modes and, subject to available clip counts, across sessions. Two human-audited
toolhead-collision intervals occur outside `printing_motion`; they remain valid
fault-active test events.

## Explicitly Superseded

- source/target division;
- the `121.44 mm/s` boundary;
- limited-target versus source-only comparison;
- the full experiment design in `asd3dp_composite_split_v2_tables.pdf`;
- every manifest under the old `splits_v2` design.

The old files remain preserved as superseded design history.

## Safety Gate

Current gate: `VALIDATED` for starting the locked 100-epoch runs. The 1-epoch
smoke metrics are diagnostic only and are not paper evidence.

Training may start only after one new Run 20 manifest is created and audited
against the canonical release manifest. The audit must confirm:

- exactly 1,000 normal mono WAV rows;
- no anomaly rows in training;
- domain-balanced sampling strata and realized counts;
- nearest-context expansion order, if used;
- no synchronized `clip_uid` duplication;
- no train/test identity leakage;
- all WAV paths exist;
- the same accepted manifest is used for all three model seeds.

The evaluation record must include the fitted Gamma parameters, the normal
scores used for fitting, and the resulting 90% quantile F1 threshold.

## Links And Evidence

- [Run 20 index](../README.md)
- [Superseded Runs 10-12 protocol](../../10_12_asd3dp_baselines/docs/EXPERIMENT_PROTOCOL.md)
- [Superseded composite split v2](../../10_12_asd3dp_baselines/splits_v2/SPLIT_LOCK.md)
- Canonical candidate manifest: `runs/07_v041_clip_selection/outputs/v041_event_clean_manifest.csv`
- [Run 20 train manifest](../splits/run20_train.csv)
- [Run 20 test manifest](../splits/run20_test.csv)
- [Run 20 split audit](../splits/split_audit.json)
