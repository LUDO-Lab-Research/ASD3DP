# ASD3DP v0.4.1 Dataset Card

## Dataset summary

ASD3DP is a four-channel operating-sound dataset for anomalous sound detection
and machine-condition monitoring in fused-filament-fabrication 3D printing. It
was recorded from an unenclosed Voron 0.2-class printer during actual machine
operation. The dataset contains normal printing and motion-only sessions,
controlled physical fault protocols, and observed machine anomaly events.

The dataset is audio-centered but process-aware. It includes time-aligned audio,
G-code resources, telemetry synchronization, phase windows, fault timelines,
session metadata, material metadata, machine-component metadata, and fixed
10-second event-clean clips.

## Motivation

Many 3D-printer subsystems produce sound at the same time: motors, belts,
pulleys, bearings or rails, fans, extruder gears, filament transport, and the
toolhead. Their acoustic signatures depend on process phase and operating
condition. A useful ASD benchmark should therefore capture both mechanical
anomalies and the operational context needed to interpret them.

## Version

- Dataset: ASD3DP
- Version: 0.4.1
- Release status: [DRAFT / PUBLIC]
- Zenodo DOI: [ZENODO_DOI]
- Associated paper: [PAPER_DOI_OR_URL]

## Recording system

| Item | Description |
|---|---|
| Machine | Unenclosed Voron 0.2-class FFF printer |
| Interface | Zoom AMS-44 |
| Microphones | Four Behringer C-2 condenser microphones |
| Positions | Front, left, right, and rear |
| Placement | Approximately 5 cm from the corresponding printer side |
| Orientation | Diagonally toward the printer |
| Sampling rate | 48 kHz |
| Nominal bit rate | 768 kb/s per mono session-channel WAV |
| Channel mapping | Defined in `audio_channels.csv` |

## Scale

| Partition | Sessions | Unique duration | 10 s clips | Clip duration |
|---|---:|---:|---:|---:|
| Normal | 19 | 13.96 h | 4,779 | 13.28 h |
| Anomaly | 22 | 14.93 h | 4,216 | 11.71 h |
| Total | 41 | 28.90 h | 8,995 | 24.99 h |

There are 164 session-channel WAV files and 115.58 channel-hours across the
four microphones.

## Anomaly groups

| Group | Top-level class | Sessions | Duration | Clips |
|---|---|---:|---:|---:|
| Loose A, B, or A+B belt conditions | Soft anomaly | 12 | 6.92 h | 2,330 |
| Extruder failure / partial nozzle obstruction | Soft anomaly | 6 | 3.05 h | 990 |
| Toolhead collision/crash | Hard anomaly | 4 | 4.97 h | 896 |

## Label definitions

- **Normal:** expected operation for the current print or motion state.
- **Soft anomaly:** degradation or instability that may allow continued
  operation but increases quality loss or failure risk.
- **Hard anomaly:** collision or near-catastrophic condition associated with
  immediate interruption, invalid output, or damage risk.

Labels reflect operational consequence rather than sound level.

## Data provenance

The release combines:

- normal loaded printing;
- no-filament motion-only operation;
- controlled belt-tension conditions;
- controlled blocked-path or G-code-based extruder/nozzle protocols;
- G-code-based toolhead-collision protocols;
- observed real-machine sounds and surrounding environmental noise.

“G-code simulation” means the G-code controlled the physical printer during a
fault-induction or motion protocol. The released waveforms are not synthesized.

## Annotation resources

The release includes:

- session and channel metadata;
- session-to-clip mappings;
- normal/anomaly clip manifests;
- clip-level event/fault labels;
- print-window membership;
- condition and protocol labels;
- session phase windows;
- G-code timing;
- audio/G-code alignment;
- synchronized telemetry;
- session phase/fault timelines;
- rule-based candidate intervals for selected non-toolhead conditions.

Rule-based candidate intervals are not automatically equivalent to verified
human event ground truth.

## Intended uses

- unsupervised and semi-supervised ASD;
- multichannel or spatial acoustic modeling;
- audio-plus-G-code and audio-plus-telemetry fusion;
- soft-versus-hard anomaly classification;
- temporal anomaly detection;
- print-phase-aware monitoring;
- condition and protocol domain-shift analysis;
- future cross-build domain-generalization research.

## Out-of-scope uses

- certified safety control;
- autonomous emergency intervention without independent safeguards;
- claims of universal performance across all FFF or Voron printers;
- treating induced protocols as exhaustive models of naturally occurring
  failures;
- random clip-level splitting that allows the same session in train and test.

## Evaluation recommendations

Use session-disjoint partitions. Report AUC and pAUC for ASD comparability, but
also include thresholded and operational metrics such as precision, recall,
F1, false alarms per hour, event-based F1, detection delay, lead time, and
performance loss across channels or operating conditions.

## Baseline status

Only the official DCASE 2025 Task 2 autoencoder baseline has been evaluated for
the associated baseline study. Additional architectures remain future work.

## Known limitations

1. One printer build in v0.4.1.
2. One primary four-microphone geometry.
3. Limited set of fault families.
4. Some faults are deliberately induced.
5. Slow/fast labels describe sessions rather than constant instantaneous speed.
6. Rule-based candidate annotations require careful interpretation.
7. Cross-build domain generalization is proposed but not established by this
   version.

## Future expansion

The planned multi-printer extension will use independently assembled Voron V0
systems. Differences in vendors, printed-part quality, materials, tolerances,
calibration, maintenance, and wear will be treated as realistic machine-domain
variation.

## Safety and ethics

Fault induction can damage equipment or create heat and fire risks. Procedures
must not be repeated without expert supervision, a documented risk assessment,
and a controlled environment. Recordings should be screened for identifiable
speech or other private information before public release.

## Platform attribution

The printer is based on the open-source VORON 0.2 / Voron Zero design by VORON
Design. ASD3DP is independent and is not produced, sponsored, or endorsed by
VORON Design.

- https://vorondesign.com/voron0.2
- https://github.com/voronDesign/Voron-0
