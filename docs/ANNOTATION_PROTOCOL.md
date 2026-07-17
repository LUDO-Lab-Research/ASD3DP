# ASD3DP Annotation Protocol

## Purpose

This protocol defines how ASD3DP labels normal operation, soft anomalies, hard
anomalies, event boundaries, provenance, confidence, and process context. It is
intended to keep session-level, clip-level, G-code-aligned, and telemetry-aligned
annotations consistent.

## Released schema and version basis

This protocol uses conceptual state names in explanatory prose. The released
CSV serialization is defined in [`ANNOTATION_SCHEMA.md`](ANNOTATION_SCHEMA.md).
The current annotation bundle is `asd3dp_v0.4.3_public_annotation_bundle`.
The bundle file inventory and hashes are published in the
[`annotation_bundle_manifest.csv`](../metadata/annotation_bundle_manifest.csv)
and
[`annotation_bundle_sha256.txt`](../metadata/annotation_bundle_sha256.txt)
metadata files.

## Label hierarchy

```text
state
├── normal
├── soft_anomaly
│   ├── belt_tension
│   │   ├── loose_a
│   │   ├── loose_b
│   │   └── loose_a_b
│   └── extrusion_instability
│       ├── blocked_path
│       ├── partial_nozzle_obstruction
│       └── developing_clog_or_extruder_failure
└── hard_anomaly
    └── collision
        └── toolhead_collision
```

The lower-level labels should reflect what is supported by the protocol and
available evidence. Do not assign a highly specific physical cause when the
recording only establishes a broader condition.

### CSV serialization

The conceptual state names and their serialized CSV `condition` values are:

| Conceptual state | CSV `condition` value |
|---|---|
| `normal` | `normal` |
| `soft_anomaly` | `soft-anomaly` |
| `hard_anomaly` | `hard-anomaly` |

Consumers should use the serialized values when parsing the released files.
The release-assigned severity mapping is:

| CSV `fault_mode` | CSV `condition` |
|---|---|
| `none` | `normal` |
| `belt_tension_abnormal` | `soft-anomaly` |
| `extruder_cogging_or_no_extrusion` | `soft-anomaly` |
| `toolhead_collision` | `hard-anomaly` |

In the current release, `extruder_cogging_or_no_extrusion` is a soft anomaly,
and `toolhead_collision` is the only hard-anomaly fault mode.

## Top-level definitions

### Normal

Expected machine behavior for the current process phase, speed, load, and
printer configuration. Loud or distinctive sound is not automatically
anomalous.

### Soft anomaly

A non-catastrophic abnormal condition that may permit continued operation but
indicates degradation, instability, reduced print quality, or increased risk of
later failure.

Examples:

- loose A belt;
- loose B belt;
- loose A+B belts;
- developing nozzle obstruction;
- intermittent extruder failure.

### Hard anomaly

A catastrophic or near-catastrophic event associated with immediate collision,
interruption, invalid output, or high mechanical-damage risk.

Current class:

- toolhead collision/crash.

## Annotation unit

The primary unit is a recording session. Fixed-duration clips inherit or derive
labels from session timelines and event intersections. A clip must remain linked
to its source session, microphone channel, and source time interval.

## Required event fields

Use the existing release schema where available. The conceptual minimum is:

| Field | Meaning |
|---|---|
| `session_id` | Source recording-session identifier |
| `channel_id` | Microphone/channel identifier |
| `start_time_s` | Event start in source-session time |
| `end_time_s` | Event end in source-session time |
| `state` | `normal`, `soft_anomaly`, or `hard_anomaly` |
| `fault_family` | Belt, extrusion/nozzle, collision, or other |
| `fault_mode` | Most specific supported label |
| `process_phase` | Heating, homing, printing, travel, motion only, etc. |
| `provenance` | Observed, controlled protocol, rule-based candidate, or unknown |
| `confidence` | High, medium, or low |
| `evidence` | Audio, G-code, telemetry, visual inspection, or combined |
| `outcome` | Continued, degraded, stopped, failed, or unknown |
| `notes` | Brief justification and ambiguity notes |

Map these concepts onto the actual CSV column names rather than creating
duplicate fields without need.

The v0.4.3 annotation CSV files do not provide an explicit per-event
`confidence` column. Fields such as `annotation_authority`, `event_evidence`,
`source`, and `condition_source_type` record provenance or generation basis;
they are not calibrated confidence scores.

## Temporal boundaries

Where the event has a gradual onset, distinguish:

- **candidate onset:** earliest plausible acoustic or process deviation;
- **clear onset:** point at which the anomaly is reliably observable;
- **end:** return to expected behavior, intervention, or session termination.

If the release schema contains only one start boundary, use the clear onset and
record earlier ambiguous evidence in notes.

### Released temporal-label authority

The release-assigned condition class and the authority of its temporal interval
are separate properties:

| `fault_mode` | `condition` | Released temporal authority |
|---|---|---|
| `none` | `normal` | `printing_motion_window` |
| `belt_tension_abnormal` | `soft-anomaly` | `gcode_rule_interval` |
| `extruder_cogging_or_no_extrusion` | `soft-anomaly` | `gcode_rule_interval` |
| `toolhead_collision` | `hard-anomaly` | `human_audio_interval` |

Normal clips are selected from printing-motion windows. Belt-tension and
extruder intervals are rule-derived candidates. Toolhead-collision intervals
use human-reviewed audio intervals. Thus, for example, an extruder clip's
release-assigned `soft-anomaly` class does not make its precise active interval
independently verified human event ground truth.

## Provenance labels

- `observed_real_machine`: naturally observed or unintentionally occurring.
- `controlled_physical_protocol`: deliberate belt, blocked-path, or collision
  protocol recorded from the physical printer.
- `rule_based_candidate`: interval generated by a rule or synchronization
  heuristic and not independently verified as event ground truth.
- `unknown`: provenance cannot be established.

Never describe controlled physical recordings as synthesized audio.

## Audio-first review

1. Review the session audio and identify candidate deviations.
2. Mark a provisional time interval and broad class.
3. Consult G-code, telemetry, phase windows, and available visual or maintenance
   evidence.
4. Refine the boundary and label.
5. Assign provenance and, when the working schema supports it, confidence.
6. Record ambiguities rather than forcing unsupported certainty.

## Clip labeling

For each 10-second clip:

1. identify its source-session interval;
2. intersect it with phase and fault timelines;
3. exclude transition-contaminated clips from event-clean subsets when required;
4. retain the mapping to all source events;
5. avoid using directory name as the only label source.

The exact inclusion threshold for a clip intersecting an anomaly must be
documented in the release code or manifest specification.

## Confidence

The following is a conceptual review scale. It is not serialized as a
per-event field in the v0.4.3 annotation CSV files.

- **High:** clear audio evidence with supporting G-code, telemetry, protocol, or
  physical observation.
- **Medium:** clear evidence in one modality with limited corroboration, or
  consistent multimodal evidence with uncertain boundaries.
- **Low:** suspicious interval with uncertain cause, timing, or class.

## Quality control

Before release:

- independently review a defined subset of sessions;
- record disagreement rates for top-level state, fault family, and boundaries;
- verify all clip-to-session mappings;
- verify that channel-aligned files share the intended timeline;
- inspect rule-based candidate intervals separately from verified events;
- document all corrections in the changelog.

If no second-annotator study has been completed, state that limitation instead
of reporting unsupported agreement figures.

## Split policy

All official splits must be session-disjoint. If future releases contain
multiple printer builds, publish both:

- within-build session-disjoint splits; and
- leave-one-build-out domain-generalization splits.
