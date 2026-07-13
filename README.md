# ASD3DP

**A real-world four-channel 3D-printer operating-sound dataset for anomalous sound detection**

<!-- Add after Zenodo publication:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21313911.svg)](https://doi.org/10.5281/zenodo.21313911)
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

> **Dataset download:** [[Temporary Google Drive](https://drive.google.com/drive/folders/1KcSe3y6K33YjB2NMadTsnkD7ZXzqbcQv?usp=sharing)]

## Why 3D-printer sound?

An FFF 3D printer is a coupled mechatronic system containing stepper motors,
belts, pulleys, bearings or linear rails, an extruder, filament transport,
cooling fans, and a rapidly moving toolhead. These parts generate overlapping
acoustic patterns whose timing and spectral content vary with motion direction,
speed, acceleration, extrusion state, print phase, mechanical condition, and
environmental noise. This makes 3D printing a useful real-world testbed for
audio-based condition monitoring.

## Dataset at a glance

| Property | ASD3DP |
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
| Unique printer-operation duration | 28.99 h |
| Four-channel summed duration | 115.97 channel-hours |
| Selected event-clean clips | 9,067 × 10 s |
| Selected-clip duration | 25.19 h |

Use `annotations/session/audio_channels.csv` as the authoritative mapping
between microphone position and numeric channel identifier.

## Recording layout

The four microphones were placed around the printer and aimed diagonally toward
the machine. Distances are approximate and refer to the nearest corresponding
printer side.

```text
                              REAR MICROPHONE
                                    ↓
          LEFT MICROPHONE  →  [ VORON V0.2 ]  ←  RIGHT MICROPHONE
                                    ↑
                             FRONT MICROPHONE

              approximately 5 cm from the corresponding printer side
```

## Dataset composition

### Overall distribution

| Partition | Sessions | Unique duration | Selected 10 s clips | Selected-clip duration |
|---|---:|---:|---:|---:|
| Normal | 19 | 14.01 h | 4,766 | 13.24 h |
| Anomaly | 22 | 14.98 h | 4,301 | 11.95 h |
| **Total** | **41** | **28.99 h** | **9,067** | **25.19 h** |

### Anomaly-family distribution

| Anomaly family | Sessions | Unique duration | Selected 10 s clips | Selected-clip duration |
|---|---:|---:|---:|---:|
| Belt tension | 12 | 6.96 h | 2,352 | 6.53 h |
| Extruder cogging or no extrusion | 6 | 3.06 h | 1,052 | 2.92 h |
| Toolhead collision | 4 | 4.97 h | 897 | 2.49 h |

### Operating-domain distribution

| State | Process domain | Condition | Speed or protocol | Sessions | Duration | Clips | Clip duration |
|---|---|---|---|---:|---:|---:|---:|
| Normal | Printing | Loaded | Slow | 12 | 10.61 h | 3,574 | 9.93 h |
| Normal | Printing | Loaded | Fast | 3 | 1.44 h | 505 | 1.40 h |
| Normal | Motion only | No filament | Slow | 1 | 0.75 h | 264 | 0.73 h |
| Normal | Motion only | No filament | Fast | 3 | 1.21 h | 423 | 1.18 h |
| Anomaly | Printing/extrusion | Extruder failure | Blocked path | 2 | 0.91 h | 294 | 0.82 h |
| Anomaly | Printing/extrusion | Extruder failure | G-code protocol, slow | 1 | 0.79 h | 282 | 0.78 h |
| Anomaly | Printing/extrusion | Extruder failure | G-code protocol, fast | 3 | 1.35 h | 476 | 1.32 h |
| Anomaly | Motion only | Belt tension | Slow | 3 | 2.56 h | 900 | 2.50 h |
| Anomaly | Motion only | Belt tension | Fast | 9 | 4.39 h | 1,452 | 4.03 h |
| Anomaly | Motion only | Toolhead collision | G-code protocol | 4 | 4.97 h | 897 | 2.49 h |

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
- extruder cogging or no extrusion.

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
├── session/
│   ├── normal/                        # normal session-channel WAV files
│   └── anomaly/                       # anomaly session-channel WAV files
├── clip/
│   ├── normal/                        # selected normal 10 s clips
│   └── anomaly/                       # selected anomalous 10 s clips
└── annotations/
    ├── session/                       # session metadata and synchronized timelines
    │   ├── gcode/                     # source G-code files and index
    │   ├── audio_gcode_sync/          # compact G-code context
    │   ├── audio_gcode_sync_full/     # full timed source G-code
    │   ├── audio_gcode_macro/         # logical macro commands
    │   ├── telemetry_sync/            # synchronized telemetry
    │   └── time_aligned_labels/       # detailed process labels
    └── clip/                          # clip labels, telemetry, and G-code context
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

Session-level files are under `annotations/session/`; clip-level files are
under `annotations/clip/`. The release additionally includes full timed G-code,
logical macro commands, synchronized telemetry, fault-active intervals,
human-reviewed collision intervals, clip telemetry, full clip G-code, and clip
process timelines.

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

The associated study reports a DCASE-style autoencoder baseline and a submitted
frozen-SSLAM nearest-neighbor system. Internal experiment directory identifiers
are not used as public system names.

## Limitations

- The current release contains one printer build and one primary microphone
  geometry.
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
<!--
## Citation

After the Zenodo record is public, replace the placeholder below and update
`CITATION.cff`.

```text
[CREATORS] ([YEAR]). ASD3DP: A Real-World Four-Channel 3D-Printer
Operating-Sound Dataset for Anomalous Sound Detection
[Data set]. Zenodo. https://doi.org/[ZENODO_DOI]
```
-->
Associated paper:

```text
[Pending]
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

- **Maintainer:** JeongSik Kim
- **Email:** jskim@ludo-lab.com
- **Issues:** Use the repository issue tracker for data, annotation, and
  reproducibility reports.
