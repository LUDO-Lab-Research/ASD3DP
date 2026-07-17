# ASD3DP v0.4.3 Annotation Schema

## Scope

This document describes the serialized labels, row grain, verified join
identifiers, provenance fields, and temporal-label authority in the ASD3DP
v0.4.3 public dataset release. It documents the released files; it does not
modify or reinterpret their values.

| Item | Value |
|---|---|
| Dataset release | `v0.4.3` |
| Sessions | 41 |
| Logical 10-second clips | 9,067 |
| Channels | `ch1`, `ch2`, `ch3`, `ch4` |

The complete annotation-file inventory, byte sizes, row counts, and SHA-256
values are in
[`metadata/annotation_bundle_manifest.csv`](../metadata/annotation_bundle_manifest.csv).
The checksum-only list is
[`metadata/annotation_bundle_sha256.txt`](../metadata/annotation_bundle_sha256.txt).
The frozen source tree also contains two non-semantic `.DS_Store` packaging
artifacts; they remain listed so the inventory matches that tree byte for byte.

## Label serialization

Protocol prose uses underscore-separated conceptual names. Released CSV files
use the following `condition` values:

| Conceptual state | Serialized CSV `condition` |
|---|---|
| `normal` | `normal` |
| `soft_anomaly` | `soft-anomaly` |
| `hard_anomaly` | `hard-anomaly` |

Parsers should treat the serialized values as the released enum and should not
assume that the conceptual spellings occur in the CSV files.

### Fault mode and severity

The v0.4.3 release-assigned mapping is:

| `fault_mode` | `condition` | `benchmark_label` |
|---|---|---|
| `none` | `normal` | `normal` |
| `belt_tension_abnormal` | `soft-anomaly` | `anomaly` |
| `extruder_cogging_or_no_extrusion` | `soft-anomaly` | `anomaly` |
| `toolhead_collision` | `hard-anomaly` | `anomaly` |

`condition` is the three-state severity field. `benchmark_label` is the binary
benchmark field, so both soft and hard anomalies serialize as `anomaly`:

```text
benchmark_label = anomaly
├── condition = soft-anomaly
└── condition = hard-anomaly
```

`extruder_cogging_or_no_extrusion` must not be interpreted as a hard anomaly.
In v0.4.3, `toolhead_collision` is the only hard-anomaly fault mode.

## Row grain and verified identifiers

The following uniqueness statements were checked against the v0.4.3 bundle.
They describe this release and are not declarations of database constraints for
future versions.

| CSV path under `annotations/` | Row grain | Rows | Verified unique identifier in v0.4.3 |
|---|---|---:|---|
| `session/condition_annotations.csv` | One released session condition | 41 | `release_session_id` |
| `session/audio_files.csv` | One session-channel audio file | 164 | (`release_session_id`, `channel`) |
| `session/fault_active_intervals.csv` | One released fault-active interval | 1,373 | `interval_id` |
| `clip/clip_plan.csv` | One logical 10-second clip | 9,067 | `clip_uid` |
| `clip/audio_clip_manifest.csv` | One logical clip-channel audio file | 36,268 | (`clip_uid`, `channel`) |
| `clip/clip_event_annotations.csv` | One logical clip-channel annotation | 36,268 | (`clip_uid`, `channel`) |
| `clip/event_clean_manifest.csv` | One logical clip-channel event-clean record | 36,268 | (`clip_uid`, `channel`) |

In each channel-expanded clip file listed above, every one of the 9,067 logical
`clip_uid` values occurs four times: once for each of `ch1`, `ch2`, `ch3`, and
`ch4`. Therefore, use (`clip_uid`, `channel`) when identifying a row in those
files. This four-row rule does not apply to row-oriented G-code or telemetry
tables, which have different row grains.

## Temporal-label authority

Condition class and temporal-interval authority answer different questions.
The class says how the released condition is categorized; the authority says
how the clip's relevant time interval was obtained.

| `fault_mode` | `condition` | `annotation_authority` / `event_evidence` | Interpretation |
|---|---|---|---|
| `none` | `normal` | `printing_motion_window` | Clip selected from a released printing-motion window |
| `belt_tension_abnormal` | `soft-anomaly` | `gcode_rule_interval` | Rule-derived temporal candidate |
| `extruder_cogging_or_no_extrusion` | `soft-anomaly` | `gcode_rule_interval` | Rule-derived temporal candidate |
| `toolhead_collision` | `hard-anomaly` | `human_audio_interval` | Human-reviewed release-audio interval |

Accordingly, the release-assigned soft/hard condition is not a statement that
every start and end time has the same evidence source. In particular, the
extruder condition is release-assigned `soft-anomaly`, while its precise active
interval is rule-derived and is not independently verified human event ground
truth.

## Provenance and confidence

The relevant released provenance fields include:

| File | Field | Role |
|---|---|---|
| `session/condition_annotations.csv` | `condition_source_type` | Session-condition source category |
| `session/fault_active_intervals.csv` | `source`, `source_rule` | Interval source and generation rule |
| `clip/clip_event_annotations.csv` | `event_evidence` | Evidence basis propagated to a clip-channel row |
| `clip/clip_event_annotations.csv` | `annotation_authority` | Temporal-label authority propagated to a clip-channel row |

No v0.4.3 annotation CSV has an explicit per-event `confidence` column. These
provenance and authority fields must not be interpreted as calibrated
confidence scores.

## Consumer guidance

- Parse released enum values exactly, including the hyphens in
  `soft-anomaly` and `hard-anomaly`.
- Use `benchmark_label` for the binary `normal`/`anomaly` task and `condition`
  for the three-state severity view.
- Keep the session identifier when constructing data splits; official splits
  must be session-disjoint.
- Keep temporal authority alongside interval-based analyses so that
  rule-derived and human-reviewed intervals are not treated as equivalent
  evidence.
- Verify downloaded annotations against the published
  [manifest](../metadata/annotation_bundle_manifest.csv) and
  [SHA-256 list](../metadata/annotation_bundle_sha256.txt).

See [`ANNOTATION_PROTOCOL.md`](ANNOTATION_PROTOCOL.md) for the annotation
procedure and conceptual definitions.
