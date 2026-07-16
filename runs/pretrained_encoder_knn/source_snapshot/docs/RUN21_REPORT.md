# Run 21: ASD3DP GenRep-Style SSLAM Baseline

Date: 2026-07-12

## Protocol

| Item | Value |
| --- | --- |
| Split | Run 20 locked split: 1,000 normal train / 600 test |
| Test labels | 300 normal / 300 anomaly |
| Near inputs | Rear `ch1`, right `ch3`, and left `ch4`, evaluated separately |
| Far input | Front microphone, `ch2` |
| Staged channel order | Stereo column 0 = selected near; column 1 = front/far |
| Audio | 10 s, 16 kHz |
| Encoder | Frozen SSLAM, layer 12 |
| Score | 1-nearest-neighbor embedding distance |
| Decision threshold | Train-normal score 95th percentile |
| Near-only view | Selected near-position SSLAM embedding |
| Residual view | Near embedding - 0.5 x front embedding |
| Source/target | Not used |
| MemMix / PRPS | Disabled |
| Repetitions | One deterministic frozen-encoder run |

## Results

### Microphone-position comparison

Front `ch2` is fixed as the far reference. The near position is changed among
rear `ch1`, right `ch3`, and left `ch4`.

| Near position | View | AUC | pAUC | F1 |
| --- | --- | ---: | ---: | ---: |
| Rear `ch1` | Near-only | 0.9873 | 0.9525 | 0.9389 |
| Rear `ch1` | Near - 0.5 x front | 0.9895 | 0.9589 | 0.9416 |
| Right `ch3` | Near-only | 0.9734 | 0.8740 | 0.8628 |
| Right `ch3` | Near - 0.5 x front | 0.9700 | 0.8736 | 0.8407 |
| Left `ch4` | Near-only | 0.9795 | 0.9043 | 0.9322 |
| Left `ch4` | Near - 0.5 x front | 0.9815 | 0.9156 | 0.9351 |

Residual subtraction helps rear and left, but not right. Rear is the strongest
near position for both near-only and residual-view scoring.

For rear, the residual view improves AUC by 0.0022 and pAUC by 0.0064 over
near-only under the same split and SSLAM embeddings.

### Fault-wise results

Each fault is compared with the same 300 normal test clips.

| System | Fault mode | AUC | pAUC |
| --- | --- | ---: | ---: |
| Rear near-only | Belt tension | 0.9892 | 0.9432 |
| Rear near-only | Extruder failure | 0.9728 | 0.9144 |
| Rear near-only | Toolhead collision | 1.0000 | 1.0000 |
| Rear - 0.5 x front | Belt tension | 0.9896 | 0.9453 |
| Rear - 0.5 x front | Extruder failure | 0.9789 | 0.9314 |
| Rear - 0.5 x front | Toolhead collision | 1.0000 | 1.0000 |

The high overall score is therefore not caused only by toolhead collision.
SSLAM L12 separates all three anomaly-session groups strongly, although
extruder failure remains the hardest fault.

### Fault AUC across microphone positions

| Near position | View | Belt | Extruder | Toolhead |
| --- | --- | ---: | ---: | ---: |
| Rear `ch1` | Near-only | 0.9892 | 0.9728 | 1.0000 |
| Rear `ch1` | Residual | 0.9896 | 0.9789 | 1.0000 |
| Right `ch3` | Near-only | 0.9781 | 0.9420 | 1.0000 |
| Right `ch3` | Residual | 0.9769 | 0.9330 | 1.0000 |
| Left `ch4` | Near-only | 0.9761 | 0.9625 | 1.0000 |
| Left `ch4` | Residual | 0.9766 | 0.9679 | 1.0000 |

Toolhead collision is perfectly ranked at every near position. The microphone
position mainly changes belt/extruder performance, with rear strongest overall.

### Why are the scores high?

- SSLAM is a large pretrained audio encoder, whereas Run 20 learns a compact
  autoencoder only from the 1,000 ASD3DP normal clips.
- A 1-NN memory bank can preserve fine session and operating-context
  differences that a reconstruction AE may smooth out.
- Toolhead collision is acoustically distinct and is perfectly ranked here.
- Belt and extruder anomaly sessions are also separated strongly in SSLAM
  embedding space.
- Normal and anomaly recordings come from different sessions. The current
  protocol prevents clip/session leakage, but it cannot prove that SSLAM uses
  fault acoustics rather than other session-specific acquisition differences.

The last point is the main claim boundary. These scores validate that the
dataset and frozen-embedding pipeline operate end to end, but they should not
be interpreted as isolated fault-causality evidence without a matched-session
or controlled context analysis.

Reported F1 uses the 95th percentile of leave-one-out train-normal scores as a
label-free decision threshold. The upstream test-label oracle F1 remains a
diagnostic output and is not used in the paper result table.

## Validation Boundary

- The Run 20 clip identities and normal/anomaly labels are reused unchanged.
- Stereo staging contains 1,000 train and 600 test files.
- Every staged file is 16 kHz, 10 s, two-channel audio.
- Front `ch2` is always far. Rear `ch1`, right `ch3`, and left `ch4` are
  evaluated independently as near.
- No source/target labels, MemMix, or PRPS selection are used.

## Links And Evidence

- Rear: [config](../config/run21_asd3dp.yaml), [staging audit](../staging/ASD3DP/stage_audit.json), [results](../out/run21_scores/result.csv), [fault metrics](../out/run21_scores/run21_fault_metrics.csv)
- Right: [config](../config/run21_right_near.yaml), [staging audit](../staging/ASD3DP_right_near/stage_audit.json), [results](../out/run21_right_scores/result.csv), [fault metrics](../out/run21_right_scores/run21_fault_metrics.csv)
- Left: [config](../config/run21_left_near.yaml), [staging audit](../staging/ASD3DP_left_near/stage_audit.json), [results](../out/run21_left_scores/result.csv), [fault metrics](../out/run21_left_scores/run21_fault_metrics.csv)
- [All-position summary](../out/run21_channel_summary.csv)
- [Label-free F1](../out/run21_label_free_f1.csv)
- [All-position fault summary](../out/run21_channel_fault_summary.csv)
- [Rear execution log](../logs/run21_20260712_134547.log)
- [Right/left execution log](../logs/run21_right_left_20260712_135839.log)
- [Run 20 split report](../../20_single_dcase_ae_baseline/docs/RUN20_REPORT.md)
