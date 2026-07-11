<!--
PRE-PUBLICATION CHECK
Replace these placeholders before making the repository public:
[ZENODO_DOI], [ZENODO_RECORD_URL], [PAPER_DOI_OR_URL], [GITHUB_OWNER],
[CONTACT_NAME], [CONTACT_EMAIL], and [FINAL_LICENSE].

Do not commit the large WAV archives to this repository unless that is an
intentional storage decision. The README assumes that the data release is
hosted on Zenodo and that GitHub hosts documentation, metadata, and code.
-->

# ASD3DP

**A real-world four-channel 3D-printer operating-sound dataset for anomalous sound detection**

<!-- Add after Zenodo publication:
[![DOI](https://zenodo.org/badge/DOI/[ZENODO_DOI].svg)](https://doi.org/[ZENODO_DOI])
-->

ASD3DP is an audio-centered dataset for anomalous sound detection (ASD) and
machine-condition monitoring in fused-filament-fabrication (FFF) 3D printing.
It contains synchronized recordings of normal operation, soft anomalies, and a
hard anomaly from a fully operational, unenclosed Voron 0.2-class printer.

The release combines four spatial microphone channels with session and clip
manifests, condition labels, G-code alignment, printer-state information,
telemetry synchronization, phase windows, and fault timelines. It is designed
for audio-only ASD, multichannel fusion, audio-plus-process-context modeling,
temporal event detection, and future cross-build domain-generalization studies.

> **Dataset download:** [Zenodo record — replace with final link]([ZENODO_RECORD_URL])

## Why 3D-printer sound?

An FFF 3D printer is a coupled mechatronic system containing stepper motors,
belts, pulleys, bearings or linear rails, an extruder, filament transport,
cooling fans, and a rapidly moving toolhead. These parts generate overlapping
acoustic patterns whose timing and spectral content vary with motion direction,
speed, acceleration, extrusion state, print phase, mechanical condition, and
environmental noise. This makes 3D printing a useful real-world testbed for
audio-based condition monitoring.

## Dataset at a glance

| Property | ASD3DP v0.4.1 |
|---|---:|
| Printer | Unenclosed Voron 0.2-class FFF printer |
| Recording interface | Zoom AMS-44 |
| Microphones | 4 × Behringer C-2 |
| Microphone positions | Front, left, right, and rear |
| Approximate distance | 5 cm from the corresponding printer side |
| Orientation | Angled diagonally toward the printer |
| Sampling rate | 48 kHz |
| Nominal audio bit rate | 768 kb/s per mono session-channel WAV |
| Recording sessions | 41 |
| Session-channel WAV files | 164 |
| Unique printer-operation duration | 28.90 h |
| Four-channel summed duration | 115.58 channel-hours |
| Selected event-clean clips | 8,995 × 10 s |
| Selected-clip duration | 24.99 h |

Use `audio_channels.csv` as the authoritative mapping between microphone
position and numeric channel identifier.

## Recording layout

The four microphones were placed around the printer and aimed diagonally toward
the machine. Distances are approximate and refer to the nearest corresponding
printer side.

```text
                         REAR MICROPHONE
                              ↘
          LEFT MICROPHONE  ↘  [ VORON V0.2 ]  ↙  RIGHT MICROPHONE
                              ↗
                         FRONT MICROPHONE

              approximately 5 cm from the corresponding printer side
```

## Dataset composition

### Overall distribution

| Partition | Sessions | Unique duration | Selected 10 s clips | Selected-clip duration |
|---|---:|---:|---:|---:|
| Normal | 19 | 13.96 h | 4,779 | 13.28 h |
| Anomaly | 22 | 14.93 h | 4,216 | 11.71 h |
| **Total** | **41** | **28.90 h** | **8,995** | **24.99 h** |

### Anomaly-family distribution

| Anomaly family | Sessions | Unique duration | Selected 10 s clips | Selected-clip duration |
|---|---:|---:|---:|---:|
| Belt tension | 12 | 6.92 h | 2,330 | 6.47 h |
| Extruder failure / nozzle obstruction | 6 | 3.05 h | 990 | 2.75 h |
| Toolhead collision | 4 | 4.97 h | 896 | 2.49 h |

### Operating-domain distribution

| State | Process domain | Condition | Speed or protocol | Sessions | Duration | Clips | Clip duration |
|---|---|---|---|---:|---:|---:|---:|
| Normal | Printing | Loaded | Slow | 12 | 10.56 h | 3,587 | 9.96 h |
| Normal | Printing | Loaded | Fast | 3 | 1.44 h | 505 | 1.40 h |
| Normal | Motion only | No filament | Slow | 1 | 0.75 h | 264 | 0.73 h |
| Normal | Motion only | No filament | Fast | 3 | 1.21 h | 423 | 1.18 h |
| Anomaly | Printing/extrusion | Extruder failure | Blocked path | 2 | 0.92 h | 288 | 0.80 h |
| Anomaly | Printing/extrusion | Extruder failure | G-code protocol, slow | 1 | 0.85 h | 265 | 0.74 h |
| Anomaly | Printing/extrusion | Extruder failure | G-code protocol, fast | 3 | 1.50 h | 437 | 1.21 h |
| Anomaly | Motion only | Belt tension | Slow | 3 | 2.56 h | 881 | 2.45 h |
| Anomaly | Motion only | Belt tension | Fast | 9 | 4.36 h | 1,449 | 4.03 h |
| Anomaly | Motion only | Toolhead collision | G-code protocol | 4 | 4.97 h | 896 | 2.49 h |

“Slow” and “fast” are session-level collection descriptors. Actual motion and
extrusion speed can vary within a session; use the synchronized telemetry for
exact speed-conditioned analysis.

## Label taxonomy

The soft/hard distinction is based on operational consequence and failure risk,
not acoustic loudness.

### Normal

Expected machine operation for the current process state, including:

- standard-speed loaded printing;
- fast loaded printing;
- no-filament motion-only operation at slow or fast settings.

### Soft anomaly

A non-catastrophic abnormal condition that may allow the printer to continue
operating but indicates degradation, instability, quality loss, or elevated
failure risk.

Current soft-anomaly conditions include:

- loose A belt;
- loose B belt;
- loose A+B belts;
- extruder failure, partial nozzle obstruction, or developing clog conditions.

### Hard anomaly

A catastrophic or near-catastrophic event associated with immediate collision,
process interruption, print invalidation, or mechanical-damage risk.

The current hard-anomaly class is:

- toolhead collision/crash.

See [`docs/ANNOTATION_PROTOCOL.md`](docs/ANNOTATION_PROTOCOL.md) for the label
definitions, provenance fields, temporal boundaries, and confidence guidance.

## Physical recordings and controlled protocols

Some anomaly conditions were induced through controlled blocked-path or
G-code-based protocols. In ASD3DP, **“G-code simulation” describes the physical
motion or fault-induction procedure; it does not mean that the audio was
synthesized**. Every released waveform was captured from the physical printer
and its surrounding environment.

## Dataset organization

```text
ASD3DP/
├── all/
│   ├── audio/                         # complete session-level audio view
│   ├── gcode/                         # public G-code files
│   ├── audio_gcode_sync/
│   │   └── by_session/                # per-session audio/G-code alignment
│   ├── telemetry_sync/
│   │   └── by_session/                # per-session synchronized telemetry
│   └── label_timeline/                # session phase/fault timelines
├── minimum/
│   ├── audio/                         # compact core session-level audio view
│   ├── gcode_annotation/              # compact aligned annotations
│   └── gcode_only_annotations/        # timing-only reduced annotations
├── clipped/
│   ├── audio_event_clean/
│   │   ├── normal/                    # selected normal 10 s clips
│   │   └── anomaly/                   # selected anomalous 10 s clips
│   └── annotations/
│       └── fault_mode_pulse_candidate_labeling.csv
└── metadata and manifest CSV files
```

Important metadata files include:

- `audio_channels.csv`
- `audio_files.csv`
- `audio_manifest.csv`
- `audio_clip_manifest.csv`
- `event_clean_manifest.csv`
- `clip_event_annotations.csv`
- `clip_print_window_membership.csv`
- `condition_annotations.csv`
- `session_profile.csv`
- `session_machine_profile.csv`
- `session_phase_windows.csv`
- `session_clip_phase_windows.csv`
- `session_clip_print_windows.csv`
- `machine_components.csv`
- `material_filament.csv`
- `firmware_software_summary.csv`
- `model_source.csv`

The rule-based `fault_mode_pulse_candidate_labeling.csv` intervals are
diagnostic candidates. They should not automatically be treated as equivalent
to independently verified human event ground truth.

## Recommended tasks

1. **Audio-only ASD:** normal versus anomalous operation.
2. **Soft/hard classification:** normal versus soft anomaly versus hard anomaly.
3. **Multichannel ASD:** early or late fusion across the four microphones.
4. **Context-aware ASD:** audio combined with G-code, telemetry, or phase state.
5. **Temporal event detection:** detection of anomaly onset and duration.
6. **Cross-condition testing:** slow/fast, loaded/motion-only, or protocol shift.
7. **Future cross-build generalization:** train on one or more Voron V0 builds
   and evaluate on an independently assembled build.

## Evaluation guidance

Split by **recording session**, not by randomly assigning 10-second clips.
Clips derived from the same source session are acoustically related, so
clip-level random splitting can leak printer, environment, and temporal context
between training and test sets.

Recommended reporting includes:

- ROC-AUC and partial AUC for comparison with ASD literature;
- precision, recall, and F1 at a stated threshold;
- soft-anomaly recall;
- hard-anomaly recall;
- event-based F1;
- false alarms per hour;
- detection delay or lead time;
- performance drop across sessions, conditions, microphones, and future builds.

## Baseline

The current baseline study evaluates **only the official DCASE 2025 Task 2
autoencoder baseline**. No results from additional architectures are claimed for
ASD3DP v0.4.1.

- DCASE 2025 Task 2:  
  https://dcase.community/challenge2025/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring

Add the exact implementation commit, channel-handling strategy, split,
normalization, seeds, AUC, and pAUC here when the experiment record is final.

## Limitations

- Version 0.4.1 contains one printer build and one primary microphone geometry.
- Controlled faults may not span the full variability of naturally developing
  field failures.
- “Slow” and “fast” are session-level descriptors rather than constant
  clip-level speeds.
- Some candidate intervals are rule-based and must remain distinguishable from
  manually verified labels.
- The release should not be interpreted as demonstrating generalization to all
  Voron or FFF printers.

## Future domain generalization

Future releases are planned to include independently assembled Voron V0
printers. Nominally similar builds can differ in component manufacturer,
printed-part material and quality, assembly tolerance, calibration, maintenance,
and wear. These differences provide a realistic basis for leave-one-build-out
and cross-build domain-generalization evaluation.

## Safety notice

ASD3DP is a research dataset, not a certified machine-safety system. Do not use
a model trained on this dataset as the sole basis for safety-critical printer
control.

Collision, obstruction, and degradation protocols can damage equipment or
create thermal and fire hazards. Do not reproduce a fault-induction procedure
without an appropriate risk assessment, suitable supervision, and a safe test
environment.

## Download and integrity

1. Download the dataset from the Zenodo record.
2. Verify the supplied checksums.
3. Read the release notes and dataset card.
4. Use the manifests rather than inferring labels from directory names alone.
5. Keep train, validation, and test sets session-disjoint.

## Citation

After the Zenodo record is public, replace the placeholder below and update
`CITATION.cff`.

```text
[CREATORS] ([YEAR]). ASD3DP: A Real-World Four-Channel 3D-Printer
Operating-Sound Dataset for Anomalous Sound Detection (Version 0.4.1)
[Data set]. Zenodo. https://doi.org/[ZENODO_DOI]
```

Associated paper:

```text
[AUTHORS]. “[PAPER TITLE].” [VENUE], [YEAR]. https://doi.org/[PAPER_DOI]
```

## Voron platform attribution

The recording platform is based on the open-source Voron Zero/V0.2 design by
VORON Design.

- VORON Design, **VORON 0.2**: https://vorondesign.com/voron0.2
- VORON Design, **Voron 0 CoreXY 3D Printer Design**:
  https://github.com/voronDesign/Voron-0

ASD3DP is an independent research dataset and is not produced, sponsored, or
endorsed by VORON Design.

## License and third-party material

**Final dataset license: `[FINAL_LICENSE]`.**

Before publication, complete a rights review for every included audio file,
annotation, G-code file, source model, diagram, and script. Third-party G-code,
models, software, and referenced design files remain subject to their own
licenses even when they are distributed alongside the dataset.

See [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md).

## Contact

- **Maintainer:** [CONTACT_NAME]
- **Email:** [CONTACT_EMAIL]
- **Issues:** Use the repository issue tracker for data, annotation, and
  reproducibility reports.
